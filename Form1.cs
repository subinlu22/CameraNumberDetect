using MvCameraControl;              // Hikrobot 카메라 SDK - 카메라 관련 모든 기능이 여기 들어있음
using OpenCvSharp;                  // OpenCV 기능 사용 - Mat, Cv2 등
using System;
using System.Collections.Generic;   // 카메라 목록을 담을 List<> 쓰려고 필요함
using System.IO;                    // 학습용 이미지 저장/폴더 생성 위해 필요함
using System.Linq;                  // 배열에서 제일 큰 값의 위치 찾을 때(argmax) 필요함
using System.Text.Json;             // label_map.json 읽을 때 필요함
using System.Threading;             // 영상을 계속 받아오는 작업을 별도 스레드로 돌리기 위해 필요함
using System.Windows.Forms;
using System.Runtime.InteropServices; // 비트맵 픽셀 데이터 직접 다룰 때 필요함 (한글 텍스트 그리기용)
using Microsoft.ML.OnnxRuntime;         // ONNX 모델 불러와서 추론 실행하는 기능
using Microsoft.ML.OnnxRuntime.Tensors; // 모델에 넣을 입력 데이터(텐서) 만드는 기능

namespace CameraNumberDetect
{
    public partial class Form1 : Form
    {
        // [카메라 종류 설정] GigE(랜선으로 연결하는 카메라) 타입만 찾겠다고 지정
        readonly DeviceTLayerType enumTLayerType = DeviceTLayerType.MvGigEDevice;

        IDevice device = null;          // 연결된 카메라를 가리키는 변수
        bool isGrabbing = false;        // "지금 영상을 계속 받아오고 있는 중인가?" 스위치
        Thread receiveThread = null;    // 영상을 계속 받아오는 작업을 담당할 별도 스레드

        // [신규] 자동촬영(학습 데이터 수집) 관련 상태값들
        bool isAutoCapturing = false;           // "지금 자동촬영 중인가?" 스위치
        string captureDigit = "";               // 지금 촬영 중인 숫자 ("1","2","3")
        int captureFrameCounter = 0;            // 프레임이 몇 번 지나갔는지 세는 카운터
        readonly int captureFrameInterval = 5;  // 몇 프레임마다 한 장씩 저장할지 (5면 대략 초당 6장 정도)
        int captureSavedCount = 0;              // 이번 촬영에서 지금까지 저장한 장수
        readonly string trainingDataFolder = Path.Combine(Application.StartupPath, "TrainingData");

        // [신규] 숫자 인식 CNN 모델 관련
        InferenceSession digitOnnxSession = null;      // ONNX 모델을 불러와서 추론 실행하는 객체
        Dictionary<string, string> digitLabelMap;      // 모델 출력 번호(0,1,2) → 실제 숫자("1","2","3") 매핑
        // [뚜껑색 → 숫자색 매핑] 숫자 색깔은 뚜껑 색깔에 100% 종속되어 있어서(스펙 고정값),
        // 카메라로 다시 "감지"할 필요 없이 이 표만 찾아보면 됨
        readonly Dictionary<string, string> lidToDigitColorMap = new Dictionary<string, string>
        {
            { "화이트", "검정" },
            { "파랑", "노랑" },
            { "빨강", "파랑" },
        };

        public Form1()
        {
            InitializeComponent();
            SDKSystem.Initialize();

            // [신규] 숫자 인식 CNN 모델 불러오기 - 프로그램 켤 때 딱 한 번만 로딩
            LoadDigitOnnxModel();

            // [창 닫힐 때 정리 작업 등록]
            this.FormClosing += Form1_FormClosing;

            // [카메라 시작 시점] - 폼이 화면에 완전히 다 뜬 다음(Load 이벤트)에 실행
            this.Load += Form1_Load;
        }

        // [신규] digit_cnn.onnx + label_map.json을 불러와서 준비해두는 함수
        // (매 프레임마다 다시 불러오면 느리니까, 프로그램 시작할 때 딱 한 번만 실행)
        private void LoadDigitOnnxModel()
        {
            try
            {
                string onnxPath = Path.Combine(Application.StartupPath, "digit_cnn.onnx");
                string labelMapPath = Path.Combine(Application.StartupPath, "label_map.json");

                digitOnnxSession = new InferenceSession(onnxPath);

                string labelMapJson = File.ReadAllText(labelMapPath);
                digitLabelMap = JsonSerializer.Deserialize<Dictionary<string, string>>(labelMapJson);

                System.Diagnostics.Debug.WriteLine("숫자 인식 CNN 모델 로딩 완료");
            }
            catch (Exception ex)
            {
                // [실패해도 프로그램은 계속 돌아가게] 모델 로딩 실패해도 카메라 자체는 써야 하니까
                // 여기서 프로그램을 죽이지 않고, 대신 나중에 추론 시도할 때 null 체크로 건너뜀
                digitOnnxSession = null;
                MessageBox.Show("숫자 인식 모델(digit_cnn.onnx) 로딩 실패: " + ex.Message +
                    "\n\ndigit_cnn.onnx, label_map.json 파일이 실행파일과 같은 폴더에 있는지 확인해줘.");
            }
        }

        // ch: 폼이 화면에 완전히 뜬 직후 자동으로 실행되는 함수
        private void Form1_Load(object sender, EventArgs e)
        {
            OpenCameraAndStartGrab();
        }

        // [신규] OpenCV의 Cv2.PutText는 한글을 못 그리는 한계가 있어서(Hershey 폰트에
        // 한글 글자 모양 자체가 없음), 대신 윈도우 System.Drawing으로 한글 텍스트를
        // 그린 다음 그 결과만 이미지 위에 얹는 방식으로 우회하는 함수.
        // color 파라미터는 OpenCV 스타일(B,G,R 순서)의 Scalar를 그대로 받음
        private void PutKoreanText(Mat img, string text, System.Drawing.Point origin, float fontSizePt, Scalar color)
        {
            // [1단계] 글자가 그려질 정확한 크기를 먼저 재봄 (너무 작으면 잘리고, 너무 크면 낭비니까)
            System.Drawing.SizeF measuredSize;
            using (var measureBmp = new System.Drawing.Bitmap(1, 1))
            using (var measureG = System.Drawing.Graphics.FromImage(measureBmp))
            using (var measureFont = new System.Drawing.Font("맑은 고딕", fontSizePt, System.Drawing.FontStyle.Bold))
            {
                measuredSize = measureG.MeasureString(text, measureFont);
            }

            int bmpWidth = (int)Math.Ceiling(measuredSize.Width) + 4;
            int bmpHeight = (int)Math.Ceiling(measuredSize.Height) + 4;

            // [2단계] 투명 배경 비트맵에 흰색으로 텍스트만 그림
            // (실제 색은 나중에 입힐 거라, 여기서는 "글자가 있는 위치"만 알면 됨)
            using (var bmp = new System.Drawing.Bitmap(bmpWidth, bmpHeight, System.Drawing.Imaging.PixelFormat.Format32bppArgb))
            {
                using (var g = System.Drawing.Graphics.FromImage(bmp))
                using (var font = new System.Drawing.Font("맑은 고딕", fontSizePt, System.Drawing.FontStyle.Bold))
                using (var brush = new System.Drawing.SolidBrush(System.Drawing.Color.White))
                {
                    g.Clear(System.Drawing.Color.Transparent);
                    g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAlias;
                    g.DrawString(text, font, brush, 2, 2);
                }

                // [3단계] 비트맵 픽셀을 직접 읽어서, 글자가 그려진(알파값 있는) 부분만
                // 원하는 색으로 이미지 위에 찍음 (배경은 그대로 안 건드림)
                var bmpData = bmp.LockBits(
                    new System.Drawing.Rectangle(0, 0, bmp.Width, bmp.Height),
                    System.Drawing.Imaging.ImageLockMode.ReadOnly,
                    System.Drawing.Imaging.PixelFormat.Format32bppArgb);

                int stride = bmpData.Stride;
                byte[] pixelData = new byte[stride * bmp.Height];
                Marshal.Copy(bmpData.Scan0, pixelData, 0, pixelData.Length);
                bmp.UnlockBits(bmpData);

                Vec3b drawColor = new Vec3b((byte)color.Val0, (byte)color.Val1, (byte)color.Val2);

                for (int y = 0; y < bmp.Height; y++)
                {
                    int imgY = origin.Y + y;
                    if (imgY < 0 || imgY >= img.Rows) continue;

                    for (int x = 0; x < bmp.Width; x++)
                    {
                        int imgX = origin.X + x;
                        if (imgX < 0 || imgX >= img.Cols) continue;

                        // [Format32bppArgb 메모리 순서] 픽셀당 4바이트: B, G, R, A 순서
                        int idx = y * stride + x * 4;
                        byte alpha = pixelData[idx + 3];
                        if (alpha > 50) // 글자 안쪽(불투명한 부분)만 색칠
                        {
                            img.Set(imgY, imgX, drawColor);
                        }
                    }
                }
            }
        }

        // [신규] 마스크(흰색 영역) 안에서 제일 큰 윤곽선 하나만 뽑아주는 공용 함수
        // (CropLidArea에서 뚜껑 영역 찾을 때 사용 중 - 계속 유지)
        private OpenCvSharp.Point[] GetBiggestContour(Mat mask)
        {
            Cv2.FindContours(mask, out OpenCvSharp.Point[][] contours, out HierarchyIndex[] hierarchy,
                RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            if (contours.Length == 0)
            {
                return null;
            }

            OpenCvSharp.Point[] biggest = contours[0];
            double biggestArea = Cv2.ContourArea(biggest);
            for (int i = 1; i < contours.Length; i++)
            {
                double area = Cv2.ContourArea(contours[i]);
                if (area > biggestArea)
                {
                    biggestArea = area;
                    biggest = contours[i];
                }
            }
            return biggest;
        }

        // [신규] 자동촬영 중인 뚜껑 이미지를 색깔+숫자별 폴더에 저장하는 함수
        // (파일명은 타임스탬프라 절대 안 겹침 - 여러 번 나눠 찍어도 안전)
        private void SaveTrainingImage(Mat croppedLid, string lidColor, string digit)
        {
            string folder = Path.Combine(trainingDataFolder, $"{lidColor}_{digit}");
            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
            }

            // [파일명] yyyyMMdd_HHmmss_fff (밀리초까지) - 연속 저장이라 초 단위론 겹칠 수 있어서 밀리초까지 씀
            string fileName = DateTime.Now.ToString("yyyyMMdd_HHmmss_fff") + ".png";
            string fullPath = Path.Combine(folder, fileName);

            Cv2.ImWrite(fullPath, croppedLid);
            captureSavedCount++;

            System.Diagnostics.Debug.WriteLine($"학습이미지 저장: {lidColor}_{digit} → {fileName} (누적 {captureSavedCount}장)");
        }

        // ch: 카메라를 찾아서 열고, 영상 받기를 시작하는 함수
        private void OpenCameraAndStartGrab()
        {
            // [카메라 목록 찾기]
            List<IDeviceInfo> deviceInfoList;
            int result = DeviceEnumerator.EnumDevices(enumTLayerType, out deviceInfoList);

            if (result != MvError.MV_OK || deviceInfoList.Count == 0)
            {
                MessageBox.Show("카메라를 찾을 수 없습니다. 연결 상태를 확인해주세요.");
                return;
            }

            IDeviceInfo deviceInfo = deviceInfoList[0];
            device = DeviceFactory.CreateDevice(deviceInfo);

            result = device.Open();
            if (result != MvError.MV_OK)
            {
                MessageBox.Show("카메라 열기 실패! 에러코드: " + result);
                return;
            }

            if (device is IGigEDevice)
            {
                IGigEDevice gigEDevice = device as IGigEDevice;
                int optimalPacketSize;
                int packetResult = gigEDevice.GetOptimalPacketSize(out optimalPacketSize);
                if (packetResult == MvError.MV_OK)
                {
                    device.Parameters.SetIntValue("GevSCPSPacketSize", (long)optimalPacketSize);
                }
            }

            device.Parameters.SetEnumValueByString("AcquisitionMode", "Continuous");
            device.Parameters.SetEnumValueByString("TriggerMode", "Off");

            device.Parameters.SetEnumValueByString("ExposureAuto", "Off");
            device.Parameters.SetFloatValue("ExposureTime", 50000);

            isGrabbing = true;
            device.StreamGrabber.StartGrabbing();

            receiveThread = new Thread(ReceiveThreadProcess);
            receiveThread.Start();
        }

        // ch: 영상을 계속 반복해서 받아오는 함수 (별도 스레드에서 실행됨)
        private void ReceiveThreadProcess()
        {
            while (isGrabbing)
            {
                IFrameOut frameOut;
                int result = device.StreamGrabber.GetImageBuffer(1000, out frameOut);

                if (result == MvError.MV_OK)
                {
                    System.Diagnostics.Debug.WriteLine(
                        "프레임 받음! 크기: " + frameOut.Image.Width + " x " + frameOut.Image.Height);

                    Mat bayerMat = Mat.FromPixelData(
                        (int)frameOut.Image.Height,
                        (int)frameOut.Image.Width,
                        MatType.CV_8UC1,
                        frameOut.Image.PixelDataPtr);

                    Cv2.MinMaxLoc(bayerMat, out double minVal, out double maxVal);
                    System.Diagnostics.Debug.WriteLine($"픽셀 값 범위: min={minVal}, max={maxVal}");

                    Mat mat = new Mat();
                    Cv2.CvtColor(bayerMat, mat, ColorConversionCodes.BayerGR2BGR);
                    Cv2.CvtColor(mat, mat, ColorConversionCodes.BGR2RGB);

                    Mat displayMat = new Mat();
                    Cv2.Resize(mat, displayMat, new OpenCvSharp.Size(mat.Width / 3, mat.Height / 3));

                    string lidColor = ClassifyLidColor(displayMat);
                    System.Diagnostics.Debug.WriteLine("판정된 색깔: " + lidColor);

                    using (Mat croppedLid = CropLidArea(displayMat))
                    {
                        // [신규] 자동촬영 중이면 - 몇 프레임마다 한 장씩 크롭된 뚜껑 이미지를 저장
                        // (마스크가 아니라 색상 원본을 저장 - CNN이 색상 정보까지 보고 학습해야 하니까)
                        if (isAutoCapturing)
                        {
                            captureFrameCounter++;
                            if (captureFrameCounter >= captureFrameInterval)
                            {
                                captureFrameCounter = 0;
                                SaveTrainingImage(croppedLid, lidColor, captureDigit);
                            }

                            // [화면 표시] 지금 촬영 중이라는 걸 빨간 글씨로 표시 - 콘솔 안 봐도 바로 알 수 있게
                            string overlayText = $"자동촬영 중: 숫자{captureDigit} ({captureSavedCount}장)";
                            PutKoreanText(displayMat, overlayText, new System.Drawing.Point(10, 10), 20f, new Scalar(0, 0, 255));
                        }

                        // [신규] CNN으로 숫자 판정 - 전처리 → 추론 → 뚜껑색/숫자색/숫자 합쳐서 화면에 표시
                        // (뚜껑이 화면에 없으면(lidColor=="없음") 판정 자체를 건너뜀 - 노이즈 오판정 방지)
                        if (lidColor != "없음")
                        {
                            using (Mat digitPatch = ExtractDigitPatchForCnn(croppedLid))
                            {
                                if (digitPatch != null)
                                {
                                    string predictedDigit = RunDigitInference(digitPatch);
                                    if (predictedDigit != null && lidToDigitColorMap.ContainsKey(lidColor))
                                    {
                                        string digitColor = lidToDigitColorMap[lidColor];
                                        string resultText = $"{lidColor} 뚜껑, {digitColor} 숫자 {predictedDigit}";

                                        // [화면 표시] 판정 결과는 초록색 글씨로 (자동촬영 빨간 글씨 아래쪽에)
                                        PutKoreanText(displayMat, resultText, new System.Drawing.Point(10, 45), 20f, new Scalar(0, 255, 0));
                                    }
                                }
                            }
                        }

                        Cv2.ImShow("뚜껑 크롭 확인", croppedLid);
                    }

                    Cv2.ImShow("OpenCV 테스트", displayMat);

                    // [신규] 키 1/2/3 - 자동촬영 토글 (같은 키 다시 누르면 중지)
                    int key = Cv2.WaitKey(1);
                    if (key == '1' || key == '2' || key == '3')
                    {
                        string pressedDigit = ((char)key).ToString();

                        if (isAutoCapturing && captureDigit == pressedDigit)
                        {
                            // 같은 숫자키를 다시 누름 → 촬영 종료
                            isAutoCapturing = false;
                            System.Diagnostics.Debug.WriteLine($"자동촬영 종료: 숫자{captureDigit}, 총 {captureSavedCount}장 저장됨");
                        }
                        else
                        {
                            // 새로 시작 (다른 숫자를 누른 경우 - 이전 숫자 촬영은 중단하고 새 숫자로 전환)
                            isAutoCapturing = true;
                            captureDigit = pressedDigit;
                            captureFrameCounter = 0;
                            captureSavedCount = 0;
                            System.Diagnostics.Debug.WriteLine($"자동촬영 시작: 숫자{captureDigit}");
                        }
                    }

                    displayMat.Dispose();
                    bayerMat.Dispose();
                    mat.Dispose();
                    device.StreamGrabber.FreeImageBuffer(frameOut);
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine("프레임 받기 실패, 에러코드: " + result);
                }
            }
        }

        // [신규] 뚜껑이 있는 영역만 사각형으로 딱 잘라내는 함수
        private Mat CropLidArea(Mat colorMat)
        {
            using (Mat gray = new Mat())
            using (Mat mask = new Mat())
            {
                Cv2.CvtColor(colorMat, gray, ColorConversionCodes.BGR2GRAY);
                Cv2.Threshold(gray, mask, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

                OpenCvSharp.Point[] biggest = GetBiggestContour(mask);
                if (biggest == null)
                {
                    return colorMat.Clone();
                }

                Rect boundingBox = Cv2.BoundingRect(biggest);
                Mat cropped = new Mat(colorMat, boundingBox);
                return cropped.Clone();
            }
        }

        // [신규] 파이썬 train_digit_cnn.py의 extract_digit_patch()를 C#으로 완전히 동일하게 재현.
        // CNN은 학습 때 본 것과 똑같은 방식으로 전처리된 입력을 받아야 정확도가 나오기 때문에,
        // 여기 로직은 파이썬 쪽이랑 한 글자도 다르면 안 됨 (숫자만 조금 바꿔도 결과가 달라질 수 있음).
        // 반환값: 숫자만 깔끔하게 뽑힌 64x64 흑백 이미지 (못 찾으면 null)
        private Mat ExtractDigitPatchForCnn(Mat croppedLidBgr, int outSize = 64)
        {
            int h = croppedLidBgr.Rows;
            int w = croppedLidBgr.Cols;

            // [원형 관심영역(ROI)] 사각형 귀퉁이(뚜껑 바깥 배경)를 아예 분석에서 제외
            Mat roiMask = new Mat(h, w, MatType.CV_8UC1, Scalar.All(0));
            int roiRadius = (int)(Math.Min(h, w) * 0.48);
            Cv2.Circle(roiMask, new OpenCvSharp.Point(w / 2, h / 2), roiRadius, Scalar.All(255), -1);

            // [배경색 추정] 원형 영역 안 픽셀들의 중간값(median)을 배경(=뚜껑 표면) 색으로 삼음
            Vec3b[] roiPixels = GetPixelsInMask(croppedLidBgr, roiMask);
            if (roiPixels.Length == 0)
            {
                roiMask.Dispose();
                return null;
            }
            double medianB = Median(roiPixels.Select(p => (double)p.Item0));
            double medianG = Median(roiPixels.Select(p => (double)p.Item1));
            double medianR = Median(roiPixels.Select(p => (double)p.Item2));

            // [배경과의 색 거리 계산] 각 픽셀이 배경색이랑 얼마나 다른지(유클리드 거리)
            Mat floatMat = new Mat();
            croppedLidBgr.ConvertTo(floatMat, MatType.CV_32FC3);
            Mat[] channels = Cv2.Split(floatMat);
            Cv2.Subtract(channels[0], Scalar.All(medianB), channels[0]);
            Cv2.Subtract(channels[1], Scalar.All(medianG), channels[1]);
            Cv2.Subtract(channels[2], Scalar.All(medianR), channels[2]);
            Cv2.Multiply(channels[0], channels[0], channels[0]);
            Cv2.Multiply(channels[1], channels[1], channels[1]);
            Cv2.Multiply(channels[2], channels[2], channels[2]);

            Mat diffSquaredSum = new Mat();
            Cv2.Add(channels[0], channels[1], diffSquaredSum);
            Cv2.Add(diffSquaredSum, channels[2], diffSquaredSum);
            Mat diff = new Mat();
            Cv2.Sqrt(diffSquaredSum, diff);

            // [원 바깥은 강제로 0(배경) 취급] - 절대 숫자로 안 잡히게
            Mat roiInverted = new Mat();
            Cv2.BitwiseNot(roiMask, roiInverted);
            diff.SetTo(Scalar.All(0), roiInverted);

            Cv2.MinMaxLoc(diff, out double minVal, out double maxVal);
            Mat diffNormalized = new Mat();
            diff.ConvertTo(diffNormalized, MatType.CV_8UC1, 255.0 / (maxVal + 1e-6));

            // [가우시안 블러] 글리터 반짝임 노이즈 완화
            Cv2.GaussianBlur(diffNormalized, diffNormalized, new OpenCvSharp.Size(5, 5), 0);

            // [이진화] Otsu 자동 임계값
            Mat mask = new Mat();
            Cv2.Threshold(diffNormalized, mask, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
            mask.SetTo(Scalar.All(0), roiInverted);

            // [모폴로지 열림] 자잘한 점 노이즈 제거 → [닫힘] 숫자 획 뭉치기
            Mat openKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(3, 3));
            Cv2.MorphologyEx(mask, mask, MorphTypes.Open, openKernel);
            Mat closeKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(7, 7));
            Cv2.MorphologyEx(mask, mask, MorphTypes.Close, closeKernel);

            // [제일 큰 윤곽선 = 숫자] 찾아서 정사각형으로 크롭
            OpenCvSharp.Point[] biggest = GetBiggestContour(mask);

            Mat result = null;
            if (biggest != null)
            {
                Rect box = Cv2.BoundingRect(biggest);
                if (box.Width > 0 && box.Height > 0)
                {
                    Mat digitCrop = new Mat(mask, box);
                    int side = Math.Max(box.Width, box.Height);
                    Mat padded = new Mat(side, side, MatType.CV_8UC1, Scalar.All(0));
                    int yOffset = (side - box.Height) / 2;
                    int xOffset = (side - box.Width) / 2;
                    digitCrop.CopyTo(new Mat(padded, new Rect(xOffset, yOffset, box.Width, box.Height)));

                    result = new Mat();
                    Cv2.Resize(padded, result, new OpenCvSharp.Size(outSize, outSize), 0, 0, InterpolationFlags.Area);
                    padded.Dispose();
                    digitCrop.Dispose();
                }
            }

            // [정리] 중간에 만든 임시 Mat들 다 해제 (메모리 누수 방지)
            roiMask.Dispose();
            roiInverted.Dispose();
            floatMat.Dispose();
            foreach (Mat ch in channels) ch.Dispose();
            diffSquaredSum.Dispose();
            diff.Dispose();
            diffNormalized.Dispose();
            mask.Dispose();
            openKernel.Dispose();
            closeKernel.Dispose();

            return result;
        }

        // [신규] 마스크(roiMask)가 255인 위치의 픽셀들만 배열로 뽑아주는 도우미 함수
        private Vec3b[] GetPixelsInMask(Mat colorMat, Mat mask)
        {
            List<Vec3b> pixels = new List<Vec3b>();
            for (int y = 0; y < colorMat.Rows; y++)
            {
                for (int x = 0; x < colorMat.Cols; x++)
                {
                    if (mask.At<byte>(y, x) == 255)
                    {
                        pixels.Add(colorMat.At<Vec3b>(y, x));
                    }
                }
            }
            return pixels.ToArray();
        }

        // [신규] 숫자들의 중간값(median)을 구해주는 도우미 함수
        // (평균값은 이상치에 약한데, 중간값은 노이즈에 더 안정적이라 배경색 추정에 사용)
        private double Median(IEnumerable<double> values)
        {
            List<double> sorted = values.OrderBy(v => v).ToList();
            int count = sorted.Count;
            if (count == 0) return 0;
            if (count % 2 == 1)
            {
                return sorted[count / 2];
            }
            return (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0;
        }

        // [신규] 전처리된 64x64 숫자 이미지를 CNN에 넣어서 1/2/3 중 뭔지 알아내는 함수
        // 반환값: 예측된 숫자 문자열 ("1"/"2"/"3"), 모델이 없거나 실패하면 null
        private string RunDigitInference(Mat digitPatch)
        {
            if (digitOnnxSession == null || digitLabelMap == null)
            {
                return null;
            }

            // [텐서 만들기] 모델 입력 형태 (배치1, 채널1, 높이64, 너비64), 0~1로 정규화
            var inputTensor = new DenseTensor<float>(new[] { 1, 1, digitPatch.Rows, digitPatch.Cols });
            for (int y = 0; y < digitPatch.Rows; y++)
            {
                for (int x = 0; x < digitPatch.Cols; x++)
                {
                    inputTensor[0, 0, y, x] = digitPatch.At<byte>(y, x) / 255f;
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            using (var results = digitOnnxSession.Run(inputs))
            {
                float[] output = results.First().AsEnumerable<float>().ToArray();

                // [argmax] 세 값(1일 확률, 2일 확률, 3일 확률) 중 제일 큰 것의 인덱스 찾기
                int predictedIndex = 0;
                float maxScore = output[0];
                for (int i = 1; i < output.Length; i++)
                {
                    if (output[i] > maxScore)
                    {
                        maxScore = output[i];
                        predictedIndex = i;
                    }
                }

                return digitLabelMap[predictedIndex.ToString()];
            }
        }

        // [신규] 뚜껑 색깔 판정 함수
        // 뚜껑이 화면에 아예 없을 때, Otsu 이진화가 배경 자체의 미세한 밝기 차이를
        // 억지로 갈라서 큰 덩어리를 "뚜껑"으로 오인하는 걸 막기 위해
        // 인식된 영역이 적절한 범위(2~25%) 안에 있을 때만 진짜 뚜껑으로 인정함
        // (실측: 빈 화면 46.2% / 진짜 뚜껑 4.8% - 카메라 시야가 뚜껑보다 훨씬 넓어서
        //  진짜 물체가 있을 때 오히려 계산되는 비율이 작게 나옴)
        private const double MinLidAreaRatioPercent = 2.0;
        private const double MaxLidAreaRatioPercent = 25.0;

        private string ClassifyLidColor(Mat colorMat)
        {
            using (Mat gray = new Mat())
            using (Mat mask = new Mat())
            using (Mat hsv = new Mat())
            {
                Cv2.CvtColor(colorMat, gray, ColorConversionCodes.BGR2GRAY);
                Cv2.Threshold(gray, mask, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

                int maskPixels = Cv2.CountNonZero(mask);
                double maskRatio = (double)maskPixels / (mask.Rows * mask.Cols) * 100;
                System.Diagnostics.Debug.WriteLine($"뚜껑으로 인식된 영역: {maskRatio:F1}%");

                // [뚜껑 없음 판정] 범위를 벗어나면(너무 작거나, 너무 크면) 뚜껑이 아닐 가능성이 높음
                if (maskRatio < MinLidAreaRatioPercent || maskRatio > MaxLidAreaRatioPercent)
                {
                    return "없음";
                }

                Cv2.CvtColor(colorMat, hsv, ColorConversionCodes.BGR2HSV);
                Scalar meanHsv = Cv2.Mean(hsv, mask);
                double avgHue = meanHsv.Val0;
                double avgSat = meanHsv.Val1;
                double avgVal = meanHsv.Val2;

                System.Diagnostics.Debug.WriteLine(
                    $"평균 색상 - Hue:{avgHue:F1}, Sat:{avgSat:F1}, Val:{avgVal:F1}");

                if (avgSat < 40)
                {
                    return "화이트";
                }
                else if (avgHue <= 25 || avgHue >= 155)
                {
                    return "빨강";
                }
                else if (avgHue >= 100 && avgHue <= 130)
                {
                    return "파랑";
                }
                else
                {
                    return "판정불가";
                }
            }
        }

        // ch: 프로그램(창)을 닫을 때 실행되는 정리 함수
        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            isGrabbing = false;
            if (receiveThread != null)
            {
                receiveThread.Join();
            }

            Cv2.DestroyAllWindows();

            if (device != null)
            {
                device.StreamGrabber.StopGrabbing();
                device.Close();
                device.Dispose();
            }

            SDKSystem.Finalize();
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }
    }
}
