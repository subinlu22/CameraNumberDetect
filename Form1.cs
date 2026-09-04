using MvCameraControl;              // Hikrobot 카메라 SDK - 카메라 관련 모든 기능이 여기 들어있음
using OpenCvSharp;                  // [신규] OpenCV 기능 사용 - Mat, Cv2 등
using System;
using System.Collections.Generic;   // 카메라 목록을 담을 List<> 쓰려고 필요함
using System.IO;                    // [신규] 템플릿 파일 저장/불러오기 위해 필요함
using System.Threading;             // 영상을 계속 받아오는 작업을 별도 스레드로 돌리기 위해 필요함
using System.Windows.Forms;

namespace CameraNumberDetect
{
    public partial class Form1 : Form
    {
        // [카메라 종류 설정] GigE(랜선으로 연결하는 카메라) 타입만 찾겠다고 지정
        readonly DeviceTLayerType enumTLayerType = DeviceTLayerType.MvGigEDevice;

        IDevice device = null;          // 연결된 카메라를 가리키는 변수
        bool isGrabbing = false;        // "지금 영상을 계속 받아오고 있는 중인가?" 스위치
        Thread receiveThread = null;    // 영상을 계속 받아오는 작업을 담당할 별도 스레드

        // [신규] 숫자 템플릿 저장소 - "1","2","3" 각각의 기준 모양(윤곽선)을 담아둠
        Dictionary<string, OpenCvSharp.Point[]> digitTemplates = new Dictionary<string, OpenCvSharp.Point[]>();
        // [신규] 템플릿 이미지 파일을 저장할 폴더 경로
        readonly string templateFolder = Path.Combine(Application.StartupPath, "DigitTemplates");

        public Form1()
        {
            InitializeComponent();
            SDKSystem.Initialize();

            // [창 닫힐 때 정리 작업 등록]
            this.FormClosing += Form1_FormClosing;

            // [카메라 시작 시점] - 폼이 화면에 완전히 다 뜬 다음(Load 이벤트)에 실행
            this.Load += Form1_Load;
        }

        // ch: 폼이 화면에 완전히 뜬 직후 자동으로 실행되는 함수
        private void Form1_Load(object sender, EventArgs e)
        {
            LoadDigitTemplates();   // [신규] 저장되어 있던 숫자 템플릿 먼저 불러오기
            OpenCameraAndStartGrab();
        }

        // [신규] 저장된 숫자 템플릿(1.png, 2.png, 3.png) 불러오는 함수
        private void LoadDigitTemplates()
        {
            if (!Directory.Exists(templateFolder))
            {
                Directory.CreateDirectory(templateFolder);
                System.Diagnostics.Debug.WriteLine("템플릿 폴더가 없어서 새로 만듦 - 아직 템플릿 없음");
                return;
            }

            string[] digitNames = { "1", "2", "3" };
            foreach (string digit in digitNames)
            {
                string path = Path.Combine(templateFolder, digit + ".png");
                if (File.Exists(path))
                {
                    using (Mat templateMask = Cv2.ImRead(path, ImreadModes.Grayscale))
                    {
                        OpenCvSharp.Point[] biggest = GetBiggestContour(templateMask);
                        if (biggest != null)
                        {
                            digitTemplates[digit] = biggest;
                            System.Diagnostics.Debug.WriteLine($"템플릿 불러옴: 숫자 {digit}");
                        }
                    }
                }
            }
        }

        // [신규] 마스크(흰색 영역) 안에서 제일 큰 윤곽선 하나만 뽑아주는 공용 함수
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

        // [신규] 지금 보이는 숫자 마스크를 템플릿(기준)으로 저장하는 함수
        private void SaveDigitTemplate(Mat digitMask, string digitName)
        {
            string path = Path.Combine(templateFolder, digitName + ".png");
            Cv2.ImWrite(path, digitMask);
            System.Diagnostics.Debug.WriteLine($"템플릿 저장됨: 숫자 {digitName} → {path}");

            OpenCvSharp.Point[] biggest = GetBiggestContour(digitMask);
            if (biggest != null)
            {
                digitTemplates[digitName] = biggest;
            }
        }

        // [신규] 지금 보이는 숫자 모양을 저장된 템플릿(1,2,3)들과 비교해서
        // 제일 비슷한 걸 찾아내는 함수
        private string RecognizeDigit(Mat digitMask)
        {
            if (digitTemplates.Count == 0)
            {
                return "템플릿 없음";
            }

            OpenCvSharp.Point[] biggest = GetBiggestContour(digitMask);
            if (biggest == null)
            {
                return "숫자 못찾음";
            }

            string bestMatch = "판정불가";
            double bestScore = double.MaxValue;

            foreach (var pair in digitTemplates)
            {
                double score = Cv2.MatchShapes(biggest, pair.Value, ShapeMatchModes.I1);
                System.Diagnostics.Debug.WriteLine($"  숫자 {pair.Key}과 비교 점수: {score:F3}");

                if (score < bestScore)
                {
                    bestScore = score;
                    bestMatch = pair.Key;
                }
            }

            // [변경] 임계값 0.3 → 10으로 상향 - 실측 점수(10~60대)가 예상보다 훨씬 커서
            // 일단 넉넉하게 잡아두고, 나중에 로그 보면서 다시 좁혀나갈 예정
            if (bestScore > 10)
            {
                return "판정불가";
            }

            return bestMatch;
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

                    Cv2.ImShow("OpenCV 테스트", displayMat);

                    string lidColor = ClassifyLidColor(displayMat);
                    System.Diagnostics.Debug.WriteLine("판정된 색깔: " + lidColor);

                    using (Mat croppedLid = CropLidArea(displayMat))
                    {
                        Cv2.ImShow("뚜껑 크롭 확인", croppedLid);

                        using (Mat digitMask = ExtractDigitMask(croppedLid, lidColor))
                        {
                            Cv2.ImShow("숫자 마스크 확인", digitMask);

                            int digitPixels = Cv2.CountNonZero(digitMask);
                            double digitRatio = (double)digitPixels / (digitMask.Rows * digitMask.Cols) * 100;
                            System.Diagnostics.Debug.WriteLine($"숫자로 인식된 영역: {digitRatio:F1}%");

                            int key = Cv2.WaitKey(1);
                            if (key == '1' || key == '2' || key == '3')
                            {
                                string digitName = ((char)key).ToString();
                                SaveDigitTemplate(digitMask, digitName);
                            }
                            else
                            {
                                string recognizedDigit = RecognizeDigit(digitMask);
                                System.Diagnostics.Debug.WriteLine("인식된 숫자: " + recognizedDigit);
                            }
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

        // [신규] 뚜껑 색깔에 따라 숫자 색상 필터를 자동으로 골라서
        // 숫자 부분만 마스크(흰색)로 뽑아내는 함수
        private Mat ExtractDigitMask(Mat croppedLid, string lidColor)
        {
            Mat hsv = new Mat();
            Cv2.CvtColor(croppedLid, hsv, ColorConversionCodes.BGR2HSV);

            Mat mask = new Mat();

            if (lidColor == "화이트")
            {
                // [변경] 고정값(90) 대신, 지금 뚜껑의 평균 밝기 대비 상대적으로 어두운 부분만 검정 숫자로 판단
                Scalar overallMean = Cv2.Mean(hsv);
                double avgVal = overallMean.Val2;
                double darkThreshold = avgVal * 0.5;
                Cv2.InRange(hsv, new Scalar(0, 0, 0), new Scalar(180, 255, darkThreshold), mask);
            }
            else if (lidColor == "파랑")
            {
                Cv2.InRange(hsv, new Scalar(15, 80, 80), new Scalar(35, 255, 255), mask);
            }
            else if (lidColor == "빨강")
            {
                Cv2.InRange(hsv, new Scalar(90, 80, 60), new Scalar(130, 255, 255), mask);
            }
            else
            {
                mask = Mat.Zeros(croppedLid.Size(), MatType.CV_8UC1);
            }

            // [신규] 모폴로지 닫힘(Closing) 연산 - 글리터 반짝이 때문에 자잘하게 쪼개진
            // 흰 점들을 가까운 것끼리 뭉쳐서 숫자 하나가 매끈한 덩어리로 보이게 만듦
            // (팽창(Dilate)으로 부풀려서 틈을 메운 다음, 침식(Erode)으로 다시 원래
            //  크기로 줄이는 방식 - 크기는 거의 그대로 유지되면서 틈만 메워짐)
            Mat kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(7, 7));
            Cv2.MorphologyEx(mask, mask, MorphTypes.Close, kernel);
            kernel.Dispose();

            hsv.Dispose();
            return mask;
        }

        // [신규] 뚜껑 색깔 판정 함수
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