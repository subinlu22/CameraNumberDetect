using MvCameraControl;              // Hikrobot 카메라 SDK - 카메라 관련 모든 기능이 여기 들어있음
using OpenCvSharp;                  // OpenCV 기능 사용 - Mat, Cv2 등
using System;
using System.Collections.Generic;   // 카메라 목록을 담을 List<> 쓰려고 필요함
using System.IO;                    // 학습용 이미지 저장/폴더 생성 위해 필요함
using System.Linq;                  // 배열에서 제일 큰 값의 위치 찾을 때(argmax) 필요함
using System.Text;                  // CSV 파일을 UTF-8로 저장할 때 필요함
using System.Text.Json;             // label_map.json 읽을 때 필요함
using System.Threading;             // 영상을 계속 받아오는 작업을 별도 스레드로 돌리기 위해 필요함
using System.Windows.Forms;
using System.Runtime.InteropServices; // 비트맵 픽셀 데이터 직접 다룰 때 필요함 (Mat→Bitmap 변환용)
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

        // [신규] 분석 대상 영역(ROI) 관련 - 화면 양쪽 가장자리에 컨베이어벨트 금속 레일이 있어서
        // 배경 학습 마스크에서도 이 구간은 아예 제외하고 분석함
        private const double RoiLeftMarginPercent = 25.0;
        private const double RoiRightMarginPercent = 25.0;

        // [신규] 배경 학습(BackgroundSubtractorMOG2) 관련 상태값들
        // - 벨트 자체가 계속 움직이면서 짜임 무늬가 스크롤되듯 지나가는 상황이라,
        //   짧게 학습하면 "매번 새로운 무늬가 나타난다"고 착각함 - 충분히 오래(여러 바퀴) 학습시켜야
        //   "이 자리는 원래 이렇게 반복적으로 밝아졌다 어두워졌다 한다"는 패턴을 제대로 익힘
        // - 그래서 고정된 시간이 아니라, 버튼으로 직접 시작/종료를 조절할 수 있게 함
        BackgroundSubtractorMOG2 bgSubtractor = null;
        bool isBackgroundLearning = false;
        System.Diagnostics.Stopwatch backgroundLearningStopwatch;

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

        // [신규] 관제(카운트) 관련 상태값들
        int totalInspectedCount = 0;   // 오늘 총 검사 수
        readonly Dictionary<string, int> colorCounts = new Dictionary<string, int>
        {
            { "화이트", 0 }, { "파랑", 0 }, { "빨강", 0 },
        };
        readonly Dictionary<string, int> digitCounts = new Dictionary<string, int>
        {
            { "1", 0 }, { "2", 0 }, { "3", 0 },
        };
        // [중복집계 방지 - 재설계] 시간(프레임 수)으로 판단하는 대신,
        // "지금 결과가 방금 전 카운트한 조합이랑 다른가"로 판단하는 방식.
        // - 같은 뚜껑이 계속 있거나, 손이 잠깐 지나갔다 같은 뚜껑으로 돌아오면 → lastCountedKey랑
        //   똑같으니까 중복 카운트 안 됨
        // - 다른 뚜껑으로 바뀌면 → 조합이 달라지니까 대기시간 없이 바로 새로 카운트됨
        //   (뚜껑을 빠르게 바꿔치기해도 안전하게 작동)
        string lastCountedKey = null;     // 마지막으로 카운트했던 조합 ("화이트_검정_2" 이런 식)
        string candidateKey = null;       // 지금 후보로 보이고 있는 조합
        int candidateStableFrames = 0;    // 그 후보가 몇 프레임 연속으로 똑같이 나왔는지
        const int stabilityThreshold = 5; // 이 프레임 수만큼 연속으로 같아야 "확실하다"고 보고 카운트함
        readonly string inspectionLogPath = Path.Combine(Application.StartupPath, "InspectionLog.csv");

        // [신규] 대시보드 UI 컨트롤들 - 전부 코드에서 직접 생성 (디자이너 파일 안 건드림)
        Panel panelInfo;
        Panel panelCameraBorder; // pictureBox1을 감싸서 테두리 색으로 양품/불량 표시할 패널
        Label lblResultHeader, lblResult, lblCaptureStatus;
        Label lblStatsHeader, lblTotalCount, lblColorCounts, lblDigitCounts, lblCropHeader;
        Label lblDigitPatchHeader; // [신규] CNN 입력 미리보기 라벨
        Label lblFilterHeader;
        Button btnFilterWhite, btnFilterBlue, btnFilterRed, btnFilterAll;
        PictureBox pictureBoxCrop;
        PictureBox pictureBoxDigitPatch; // [신규] CNN이 실제로 보는 64x64 입력 이미지 미리보기
        Button btnExit; // [신규] 프로그램 종료 버튼
        Button btnLearnBackground; // [신규] 배경 학습 시작/종료 버튼
        Button btnResetCount; // [신규] 오늘 통계(카운트) 리셋 버튼

        // [신규] 검사 기준(필터) - 이 색깔이 아니면 불량(빨간 테두리)으로 표시
        // null이면 "전체 허용" 상태 (기준 없음, 중립 테두리)
        string expectedLidColor = null;

        public Form1()
        {
            InitializeComponent();

            // [신규] 대시보드 UI 구성 - OpenCV 창 대신 폼 안에 예쁘게 배치
            BuildDashboardUi();

            SDKSystem.Initialize();

            // [신규] 숫자 인식 CNN 모델 불러오기 - 프로그램 켤 때 딱 한 번만 로딩
            LoadDigitOnnxModel();

            // [신규] 검사 기록 CSV 파일 준비 (없으면 헤더 줄과 함께 새로 만듦)
            InitializeInspectionLog();

            // [신규] 키보드 입력을 폼이 직접 받도록 설정 (예전엔 OpenCV 창이 받았는데,
            // 이제 OpenCV 창이 없어지니까 폼 자체에서 1/2/3 키를 받아야 함)
            this.KeyPreview = true;
            this.KeyDown += Form1_KeyDown;

            // [창 닫힐 때 정리 작업 등록]
            this.FormClosing += Form1_FormClosing;

            // [카메라 시작 시점] - 폼이 화면에 완전히 다 뜬 다음(Load 이벤트)에 실행
            this.Load += Form1_Load;
        }

        // [신규] 대시보드 UI를 코드로 직접 구성하는 함수
        // (디자이너 파일을 직접 손대지 않고, 기존 pictureBox1만 재활용 + 나머지는 여기서 새로 추가)
        private void BuildDashboardUi()
        {
            // [전체 창 설정] 고정 크기의 다크톤 대시보드
            this.Text = "산업용카메라 검사 시스템";
            this.ClientSize = new System.Drawing.Size(1200, 700);
            this.StartPosition = FormStartPosition.CenterScreen;
            this.FormBorderStyle = FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.BackColor = System.Drawing.Color.FromArgb(30, 30, 30);

            // [왼쪽 - 실시간 카메라 화면] pictureBox1을 테두리 패널 안에 넣어서
            // 테두리 색깔로 양품(초록)/불량(빨강)을 직관적으로 보여줄 수 있게 함
            panelCameraBorder = new Panel
            {
                Location = new System.Drawing.Point(20, 20),
                Size = new System.Drawing.Size(760, 610),
                BackColor = System.Drawing.Color.FromArgb(70, 70, 70), // 기본은 중립 회색
            };
            this.Controls.Add(panelCameraBorder);

            pictureBox1.Parent = panelCameraBorder; // 기존 pictureBox1을 이 패널 안으로 옮김
            pictureBox1.Location = new System.Drawing.Point(8, 8); // 8px 여백 = 테두리 두께
            pictureBox1.Size = new System.Drawing.Size(744, 594);
            pictureBox1.SizeMode = PictureBoxSizeMode.Zoom; // 비율 유지하면서 꽉 채움 (찌그러짐 방지)
            pictureBox1.BackColor = System.Drawing.Color.Black;

            // [오른쪽 - 정보 패널] 판정결과 + 통계 + 크롭 미리보기를 담을 카드
            panelInfo = new Panel
            {
                Location = new System.Drawing.Point(800, 20),
                Size = new System.Drawing.Size(360, 610),
                BackColor = System.Drawing.Color.FromArgb(45, 45, 48),
            };
            this.Controls.Add(panelInfo);

            // [신규] 검사 기준 설정 - 버튼 누르면 그 색깔만 "양품"으로 인정, 나머지는 불량 처리
            lblFilterHeader = CreateLabel("검사 기준 설정", 10, 10, 12f, true, System.Drawing.Color.White);

            btnFilterWhite = CreateFilterButton("화이트", 10, 35);
            btnFilterBlue = CreateFilterButton("파랑", 97, 35);
            btnFilterRed = CreateFilterButton("빨강", 184, 35);
            btnFilterAll = CreateFilterButton("전체", 271, 35);

            btnFilterWhite.Click += (s, e) => SetExpectedColor("화이트");
            btnFilterBlue.Click += (s, e) => SetExpectedColor("파랑");
            btnFilterRed.Click += (s, e) => SetExpectedColor("빨강");
            btnFilterAll.Click += (s, e) => SetExpectedColor(null);

            lblResultHeader = CreateLabel("판정 결과", 10, 85, 12f, true, System.Drawing.Color.White);

            lblResult = CreateLabel("대기 중", 10, 115, 18f, true, System.Drawing.Color.LightGreen);
            lblResult.Size = new System.Drawing.Size(340, 70);

            lblCaptureStatus = CreateLabel("", 10, 195, 11f, true, System.Drawing.Color.OrangeRed);

            lblStatsHeader = CreateLabel("오늘 통계", 10, 235, 12f, true, System.Drawing.Color.White);

            // [신규] 통계 리셋 버튼 - 오늘 통계 헤더 옆에 작게 배치
            btnResetCount = new Button
            {
                Text = "리셋",
                Location = new System.Drawing.Point(280, 231),
                Size = new System.Drawing.Size(60, 24),
                Font = new System.Drawing.Font("맑은 고딕", 9f),
                ForeColor = System.Drawing.Color.White,
                BackColor = System.Drawing.Color.FromArgb(70, 70, 70),
                FlatStyle = FlatStyle.Flat,
            };
            btnResetCount.Click += (s, e) => ResetCounts();

            lblTotalCount = CreateLabel("오늘 검사: 0건", 10, 270, 11f, false, System.Drawing.Color.Gainsboro);
            lblColorCounts = CreateLabel("화이트 0   파랑 0   빨강 0", 10, 300, 11f, false, System.Drawing.Color.Gainsboro);
            lblDigitCounts = CreateLabel("1번 0   2번 0   3번 0", 10, 330, 11f, false, System.Drawing.Color.Gainsboro);

            lblCropHeader = CreateLabel("뚜껑 크롭 미리보기", 10, 375, 11f, true, System.Drawing.Color.White);
            lblDigitPatchHeader = CreateLabel("CNN 입력", 250, 375, 11f, true, System.Drawing.Color.White);

            pictureBoxCrop = new PictureBox
            {
                Location = new System.Drawing.Point(10, 405),
                Size = new System.Drawing.Size(230, 150),
                SizeMode = PictureBoxSizeMode.Zoom,
                BackColor = System.Drawing.Color.Black,
                BorderStyle = BorderStyle.FixedSingle,
            };

            // [신규] CNN이 실제로 입력받는 64x64 흑백 이미지를 그대로 보여주는 미리보기
            // - 여기서 숫자 모양이 제대로 보이면 추출은 잘 된 것 → 모델(재학습) 쪽 문제
            // - 여기서부터 이미 뭉개져 있으면 추출 과정(ExtractDigitPatchForCnn) 쪽 문제
            // (크롭 미리보기 옆에 나란히 배치 - 세로로 쌓으면 패널 높이를 넘어가서 화면에 안 잡힘)
            pictureBoxDigitPatch = new PictureBox
            {
                Location = new System.Drawing.Point(250, 405),
                Size = new System.Drawing.Size(90, 90),
                SizeMode = PictureBoxSizeMode.Zoom,
                BackColor = System.Drawing.Color.Black,
                BorderStyle = BorderStyle.FixedSingle,
            };

            panelInfo.Controls.Add(lblFilterHeader);
            panelInfo.Controls.Add(btnFilterWhite);
            panelInfo.Controls.Add(btnFilterBlue);
            panelInfo.Controls.Add(btnFilterRed);
            panelInfo.Controls.Add(btnFilterAll);
            panelInfo.Controls.Add(lblResultHeader);
            panelInfo.Controls.Add(lblResult);
            panelInfo.Controls.Add(lblCaptureStatus);
            panelInfo.Controls.Add(lblStatsHeader);
            panelInfo.Controls.Add(btnResetCount);
            panelInfo.Controls.Add(lblTotalCount);
            panelInfo.Controls.Add(lblColorCounts);
            panelInfo.Controls.Add(lblDigitCounts);
            panelInfo.Controls.Add(lblCropHeader);
            panelInfo.Controls.Add(pictureBoxCrop);
            panelInfo.Controls.Add(lblDigitPatchHeader);
            panelInfo.Controls.Add(pictureBoxDigitPatch);

            // [초기 상태] "전체" 선택된 상태로 시작 (기준 없음)
            SetExpectedColor(null);

            // [신규] 하단 종료 버튼 - X 버튼 대신 안전하게 종료할 수 있는 버튼
            // (this.Close()를 호출하면 FormClosing 이벤트가 정상적으로 실행되면서 종료됨)
            btnExit = new Button
            {
                Text = "프로그램 종료",
                Location = new System.Drawing.Point(1090, 650),
                Size = new System.Drawing.Size(110, 35),
                Font = new System.Drawing.Font("맑은 고딕", 10f, System.Drawing.FontStyle.Bold),
                ForeColor = System.Drawing.Color.White,
                BackColor = System.Drawing.Color.FromArgb(150, 40, 40),
                FlatStyle = FlatStyle.Flat,
            };
            btnExit.Click += (s, e) => this.Close();
            this.Controls.Add(btnExit);

            // [신규] 배경 학습 시작/종료 버튼 - 누르는 순간부터 배경을 학습하고,
            // 다시 누르면 학습을 멈추고 그 배경 정보로 판정을 시작함
            btnLearnBackground = new Button
            {
                Text = "배경 학습 시작",
                Location = new System.Drawing.Point(950, 650),
                Size = new System.Drawing.Size(130, 35),
                Font = new System.Drawing.Font("맑은 고딕", 10f, System.Drawing.FontStyle.Bold),
                ForeColor = System.Drawing.Color.White,
                BackColor = System.Drawing.Color.FromArgb(40, 90, 150),
                FlatStyle = FlatStyle.Flat,
            };
            btnLearnBackground.Click += (s, e) => ToggleBackgroundLearning();
            this.Controls.Add(btnLearnBackground);
        }

        // [신규] 배경 학습 시작/종료를 토글하는 함수
        private void ToggleBackgroundLearning()
        {
            if (!isBackgroundLearning)
            {
                // [학습 시작] - history를 넉넉하게 잡아서 오래(여러 바퀴) 학습해도 다 기억하게 함
                bgSubtractor = BackgroundSubtractorMOG2.Create(2000, 16, true);
                isBackgroundLearning = true;
                backgroundLearningStopwatch = System.Diagnostics.Stopwatch.StartNew();
                btnLearnBackground.Text = "배경 학습 종료";
            }
            else
            {
                // [학습 종료] - 이 시점부터 판정 모드로 전환, 배경 모델은 더 이상 업데이트 안 됨
                isBackgroundLearning = false;
                btnLearnBackground.Text = "배경 학습 시작";
            }
        }

        // [신규] 오늘 통계(카운트)를 전부 0으로 되돌리는 함수
        // 중복집계 방지용 상태값(lastCountedKey 등)도 같이 초기화해야, 리셋 직후에
        // "방금 전에 이미 카운트했던 뚜껑"으로 잘못 기억해서 새 카운트를 막는 일이 없음
        private void ResetCounts()
        {
            totalInspectedCount = 0;
            colorCounts["화이트"] = 0;
            colorCounts["파랑"] = 0;
            colorCounts["빨강"] = 0;
            digitCounts["1"] = 0;
            digitCounts["2"] = 0;
            digitCounts["3"] = 0;
            lastCountedKey = null;
            candidateKey = null;
            candidateStableFrames = 0;

            lblTotalCount.Text = "오늘 검사: 0건";
            lblColorCounts.Text = "화이트 0   파랑 0   빨강 0";
            lblDigitCounts.Text = "1번 0   2번 0   3번 0";
        }

        // [신규] 검사 기준 버튼 하나 만드는 도우미 함수
        private Button CreateFilterButton(string text, int x, int y)
        {
            return new Button
            {
                Text = text,
                Location = new System.Drawing.Point(x, y),
                Size = new System.Drawing.Size(79, 32),
                Font = new System.Drawing.Font("맑은 고딕", 9f, System.Drawing.FontStyle.Bold),
                ForeColor = System.Drawing.Color.White,
                BackColor = System.Drawing.Color.FromArgb(60, 60, 60),
                FlatStyle = FlatStyle.Flat,
            };
        }

        // [신규] 검사 기준(필터) 색깔을 바꾸고, 버튼 강조 표시도 같이 갱신하는 함수
        private void SetExpectedColor(string color)
        {
            expectedLidColor = color;

            System.Drawing.Color selectedColor = System.Drawing.Color.FromArgb(90, 90, 200); // 선택된 버튼 강조색
            System.Drawing.Color normalColor = System.Drawing.Color.FromArgb(60, 60, 60);     // 평상시 버튼색

            btnFilterWhite.BackColor = (color == "화이트") ? selectedColor : normalColor;
            btnFilterBlue.BackColor = (color == "파랑") ? selectedColor : normalColor;
            btnFilterRed.BackColor = (color == "빨강") ? selectedColor : normalColor;
            btnFilterAll.BackColor = (color == null) ? selectedColor : normalColor;
        }

        // [신규] 라벨 하나 만드는 반복 작업을 줄여주는 도우미 함수
        private Label CreateLabel(string text, int x, int y, float fontSize, bool bold, System.Drawing.Color color)
        {
            return new Label
            {
                Text = text,
                Location = new System.Drawing.Point(x, y),
                AutoSize = false,
                Size = new System.Drawing.Size(340, 25),
                Font = new System.Drawing.Font("맑은 고딕", fontSize,
                    bold ? System.Drawing.FontStyle.Bold : System.Drawing.FontStyle.Regular),
                ForeColor = color,
                BackColor = System.Drawing.Color.Transparent,
            };
        }

        // [신규] 백그라운드 스레드(ReceiveThreadProcess)에서 UI 컨트롤을 안전하게 갱신하기 위한 도우미
        // (WinForms는 UI 컨트롤을 만든 스레드가 아닌 다른 스레드에서 직접 건드리면 안 되는 규칙이 있음)
        // [중요] Invoke(동기, 결과 나올 때까지 기다림) 대신 BeginInvoke(비동기, 던져놓고 바로 넘어감)를 씀.
        // Invoke를 쓰면 "창 닫기 눌러서 UI 스레드가 receiveThread.Join()으로 기다리는 중" +
        // "receiveThread는 Invoke로 UI 스레드가 처리해주길 기다리는 중"이 동시에 벌어져서
        // 서로가 서로를 기다리는 교착상태(deadlock)로 프로그램이 멈춰버리는 문제가 있었음
        private void UpdateUi(Action action)
        {
            if (this.IsHandleCreated && !this.IsDisposed)
            {
                try
                {
                    this.BeginInvoke(action);
                }
                catch (ObjectDisposedException)
                {
                    // 폼이 막 닫히는 중이면 그냥 무시 (프로그램 종료 시점의 정상적인 상황)
                }
                catch (InvalidOperationException)
                {
                    // 폼 핸들이 이미 파괴된 상태에서 호출된 경우도 무시
                }
            }
        }

        // [신규] 키보드 1/2/3 - 자동촬영 토글 (예전엔 OpenCV 창의 Cv2.WaitKey가 받던 걸
        // 이제 OpenCV 창이 없어져서 폼 자체의 키 이벤트로 옮김)
        private void Form1_KeyDown(object sender, KeyEventArgs e)
        {
            string pressedDigit = null;
            if (e.KeyCode == Keys.D1 || e.KeyCode == Keys.NumPad1) pressedDigit = "1";
            else if (e.KeyCode == Keys.D2 || e.KeyCode == Keys.NumPad2) pressedDigit = "2";
            else if (e.KeyCode == Keys.D3 || e.KeyCode == Keys.NumPad3) pressedDigit = "3";

            if (pressedDigit == null)
            {
                return;
            }

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

        // [신규] 검사 기록용 CSV 파일 준비 - 파일이 없으면 헤더 줄과 함께 새로 만듦
        // (있으면 그대로 두고 이어서 씀 - 프로그램 껐다 켜도 기록이 계속 쌓이게)
        private void InitializeInspectionLog()
        {
            if (!File.Exists(inspectionLogPath))
            {
                File.WriteAllText(inspectionLogPath, "시간,뚜껑색,숫자색,숫자\n", Encoding.UTF8);
            }
        }

        // [신규] 카운트 1건 추가 + CSV에 한 줄 기록하는 함수
        // (같은 뚜껑 중복집계 방지 체크는 호출하는 쪽(ReceiveThreadProcess)에서 이미 하고 옴)
        private void RecordInspection(string lidColor, string digitColor, string digit)
        {
            totalInspectedCount++;
            if (colorCounts.ContainsKey(lidColor))
            {
                colorCounts[lidColor]++;
            }
            if (digitCounts.ContainsKey(digit))
            {
                digitCounts[digit]++;
            }

            string logLine = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss},{lidColor},{digitColor},{digit}\n";
            File.AppendAllText(inspectionLogPath, logLine, Encoding.UTF8);
        }

        // ch: 폼이 화면에 완전히 뜬 직후 자동으로 실행되는 함수
        private void Form1_Load(object sender, EventArgs e)
        {
            OpenCameraAndStartGrab();
        }

        // [신규] OpenCV Mat을 화면(PictureBox)에 표시할 수 있는 Bitmap으로 변환하는 함수
        // (한 줄씩 그대로 복사하는 방식이라 색이 안 틀어지고 빠름 - Mat은 BGR 3채널이어야 함)
        private System.Drawing.Bitmap MatToBitmap(Mat mat)
        {
            int width = mat.Cols;
            int height = mat.Rows;

            var bitmap = new System.Drawing.Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);
            var bmpData = bitmap.LockBits(
                new System.Drawing.Rectangle(0, 0, width, height),
                System.Drawing.Imaging.ImageLockMode.WriteOnly,
                System.Drawing.Imaging.PixelFormat.Format24bppRgb);

            int matStep = (int)mat.Step();
            int rowBytes = width * 3;
            IntPtr matDataPtr = mat.Data;
            byte[] rowBuffer = new byte[rowBytes];

            // [한 줄씩 복사] Mat이랑 Bitmap이 한 줄(row)당 차지하는 바이트 수(step/stride)가
            // 다를 수 있어서, 통째로 한 번에 복사하지 않고 줄 단위로 나눠서 복사함
            for (int y = 0; y < height; y++)
            {
                IntPtr srcRow = IntPtr.Add(matDataPtr, y * matStep);
                IntPtr dstRow = IntPtr.Add(bmpData.Scan0, y * bmpData.Stride);
                Marshal.Copy(srcRow, rowBuffer, 0, rowBytes);
                Marshal.Copy(rowBuffer, 0, dstRow, rowBytes);
            }

            bitmap.UnlockBits(bmpData);
            return bitmap;
        }

        // [신규] 마스크(흰색 영역) 안에서 제일 큰 윤곽선 하나만 뽑아주는 공용 함수
        // (숫자 모양 추출(ExtractDigitPatchForCnn)에서 사용 중 - 숫자는 원형이 아니라서
        //  원형도 체크 없이 그냥 제일 큰 것만 찾음. 뚜껑 영역 찾기는 GetBiggestCircularContour로 분리됨)
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

        // [신규] 마스크 안에서 "원형에 가장 가까운" 가장 큰 윤곽선을 찾는 함수
        // (CropLidArea 전용 - 뚜껑은 원형이니까, 반사광처럼 삐뚤빼뚤하거나 길쭉한 덩어리는
        //  아무리 커도 애초에 후보에서 제외함. 숫자 추출(GetBiggestContour)엔 안 씀 - 숫자는 안 둥그니까)
        // 원형도(circularity) = 4π×면적 / 둘레² → 완벽한 원이면 1.0, 찌그러질수록 0에 가까워짐
        //
        // [신규] 원형도는 원본 윤곽선이 아니라 볼록 껍질(convex hull)로 감싼 뒤 계산함.
        // 벨트 텍스처 때문에 마스크 경계가 톱니처럼 삐죽삐죽하게 나오는데, 이걸 원본 그대로
        // 쓰면 실제로는 둥근 뚜껑이어도 둘레만 부풀려져서 원형도가 뚝 떨어지는 문제가 있었음.
        // hull로 감싸면 삐죽삐죽한 부분은 무시되고 전체적인 큰 형태만 남아서 훨씬 정확해짐
        private OpenCvSharp.Point[] GetBiggestCircularContour(Mat mask, double minCircularity = 0.6)
        {
            Cv2.FindContours(mask, out OpenCvSharp.Point[][] contours, out HierarchyIndex[] hierarchy,
                RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            OpenCvSharp.Point[] best = null;
            double bestArea = 0;

            foreach (var contour in contours)
            {
                OpenCvSharp.Point[] hull = Cv2.ConvexHull(contour);
                double area = Cv2.ContourArea(hull);
                double perimeter = Cv2.ArcLength(hull, true);
                if (perimeter <= 0)
                {
                    continue;
                }

                double circularity = 4 * Math.PI * area / (perimeter * perimeter);
                System.Diagnostics.Debug.WriteLine($"윤곽선 후보 - 면적(hull):{area:F0}, 원형도(hull기준):{circularity:F2}");

                // [원형 아닌 건 후보에서 제외] 아무리 커도 둥글지 않으면 뚜껑이 아님
                if (circularity < minCircularity)
                {
                    continue;
                }

                if (area > bestArea)
                {
                    bestArea = area;
                    best = contour;
                }
            }

            return best;
        }

        // [신규] 분석할 가운데 구간(ROI)의 사각형 범위를 계산해주는 함수
        // (화면 전체 중 좌우 가장자리 - 컨베이어벨트 금속 레일 위치 - 를 제외한 영역)
        private Rect GetAnalysisRoi(Mat colorMat)
        {
            int leftX = (int)(colorMat.Width * RoiLeftMarginPercent / 100.0);
            int rightX = (int)(colorMat.Width * (100.0 - RoiRightMarginPercent) / 100.0);
            int roiWidth = rightX - leftX;
            return new Rect(leftX, 0, roiWidth, colorMat.Height);
        }

        // [신규] 배경 학습 모델(bgSubtractor)로 "학습된 배경과 다르게 새로 나타난 부분(=뚜껑)"만
        // 뽑아낸 흑백 마스크를 만들어주는 함수
        // learningRate=0으로 고정 - 학습 단계에서 이미 배운 배경을 그대로 쓰고,
        // 판정 중에는 배경을 더 이상 업데이트하지 않음
        private Mat GetForegroundMask(Mat colorMat)
        {
            Mat fgMask = new Mat();
            bgSubtractor.Apply(colorMat, fgMask, 0);

            // [신규 - 완화] detectShadows=true라 그림자는 회색(127)으로 나오는데,
            // 뚜껑 표면 자체의 어두운 음영 부분까지 "그림자"로 오분류되어 잘려나가는 문제가
            // 의심됨(진짜 뚜껑 크기가 예상보다 훨씬 작게 잡히는 현상과 일치) → 기준을 낮춰서
            // 그림자(127)로 분류된 부분도 일단 전경에 포함시킴 (기준 200 → 50)
            Cv2.Threshold(fgMask, fgMask, 50, 255, ThresholdTypes.Binary);

            // [레일 구간 강제 제외] 양쪽 금속 레일 부분은 배경 학습이 불완전해도 절대 뚜껑으로
            // 안 잡히게, 아예 그 부분을 검정(배경)으로 덮어씀
            Rect roi = GetAnalysisRoi(colorMat);
            if (roi.X > 0)
            {
                Cv2.Rectangle(fgMask, new Rect(0, 0, roi.X, fgMask.Rows), Scalar.All(0), -1);
            }
            int rightStart = roi.X + roi.Width;
            if (rightStart < fgMask.Cols)
            {
                Cv2.Rectangle(fgMask, new Rect(rightStart, 0, fgMask.Cols - rightStart, fgMask.Rows), Scalar.All(0), -1);
            }

            // [노이즈 정리] 자잘한 점 노이즈 제거(열림) → 뚜껑 영역 안 구멍 메우기(닫힘)
            Mat openKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(3, 3));
            Cv2.MorphologyEx(fgMask, fgMask, MorphTypes.Open, openKernel);
            Mat closeKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(9, 9));
            Cv2.MorphologyEx(fgMask, fgMask, MorphTypes.Close, closeKernel);
            openKernel.Dispose();
            closeKernel.Dispose();

            return fgMask;
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
            // [확정값] MVS에서 실시간으로 조절해서 화면이 제일 선명하고 적당히 밝았던 값 그대로 반영
            device.Parameters.SetFloatValue("ExposureTime", 37021);

            device.Parameters.SetEnumValueByString("GainAuto", "Off");
            int gainResult = device.Parameters.SetFloatValue("Gain", 1.6155f);
            System.Diagnostics.Debug.WriteLine(
                gainResult == MvError.MV_OK ? "게인 설정 성공" : $"게인 설정 실패! 에러코드: {gainResult}");

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

                    // [신규] 아직 배경 학습을 한 번도 안 했으면 - 판정 자체가 불가능하니 안내만 하고 건너뜀
                    if (bgSubtractor == null)
                    {
                        System.Drawing.Bitmap waitBitmap = MatToBitmap(displayMat);
                        UpdateUi(() =>
                        {
                            var oldMain = pictureBox1.Image;
                            pictureBox1.Image = waitBitmap;
                            oldMain?.Dispose();
                            lblResult.Text = "[배경 학습 시작] 버튼을\n먼저 눌러주세요";
                        });
                        displayMat.Dispose();
                        bayerMat.Dispose();
                        mat.Dispose();
                        device.StreamGrabber.FreeImageBuffer(frameOut);
                        continue;
                    }

                    // [신규] 배경 학습 중이면 - 판정은 건너뛰고, MOG2에 이 프레임만 흘려보내면서
                    // 화면엔 경과 시간만 보여줌 (버튼을 다시 누를 때까지 계속 학습)
                    if (isBackgroundLearning)
                    {
                        using (Mat learnMask = new Mat())
                        {
                            bgSubtractor.Apply(displayMat, learnMask); // learningRate 기본값(자동)으로 학습
                        }

                        double elapsedSeconds = backgroundLearningStopwatch.Elapsed.TotalSeconds;
                        System.Drawing.Bitmap learningBitmap = MatToBitmap(displayMat);
                        string learningText = $"배경 학습 중... ({elapsedSeconds:F0}초 경과)\n벨트가 충분히 여러 바퀴 돌면\n[배경 학습 종료] 버튼을 눌러주세요";

                        UpdateUi(() =>
                        {
                            var oldMain = pictureBox1.Image;
                            pictureBox1.Image = learningBitmap;
                            oldMain?.Dispose();
                            lblResult.Text = learningText;
                        });

                        displayMat.Dispose();
                        bayerMat.Dispose();
                        mat.Dispose();
                        device.StreamGrabber.FreeImageBuffer(frameOut);
                        continue; // 판정 로직은 건너뛰고 다음 프레임으로
                    }

                    string lidColor = ClassifyLidColor(displayMat);
                    System.Diagnostics.Debug.WriteLine("판정된 색깔: " + lidColor);

                    // [화면에 표시할 텍스트들] 이번 프레임에서 뭘 보여줄지 미리 준비해뒀다가
                    // 마지막에 한 번에 UI 스레드로 넘겨서 갱신함
                    string autoCaptureStatusText = "";
                    string resultText = "대기 중 (뚜껑을 카메라 앞에 놓아주세요)";

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

                            autoCaptureStatusText = $"● 자동촬영 중: 숫자{captureDigit} ({captureSavedCount}장)";
                        }

                        // [신규] CNN 입력 미리보기용 비트맵 - digitPatch가 없으면(뚜껑 없음 등) null로 둠
                        System.Drawing.Bitmap digitPatchBitmap = null;

                        // [신규] CNN으로 숫자 판정 - 전처리 → 추론 → 뚜껑색/숫자색/숫자 합쳐서 화면에 표시
                        // (뚜껑이 화면에 없으면(lidColor=="없음") 판정 자체를 건너뜀 - 노이즈 오판정 방지)
                        if (lidColor != "없음")
                        {
                            using (Mat digitPatch = ExtractDigitPatchForCnn(croppedLid))
                            {
                                if (digitPatch != null)
                                {
                                    // [신규] CNN 입력 미리보기 - digitPatch는 흑백(1채널)이라
                                    // MatToBitmap(3채널 전제)에 넣기 전에 컬러로 변환해줘야 함
                                    using (Mat digitPatchBgr = new Mat())
                                    {
                                        Cv2.CvtColor(digitPatch, digitPatchBgr, ColorConversionCodes.GRAY2BGR);
                                        digitPatchBitmap = MatToBitmap(digitPatchBgr);
                                    }

                                    string predictedDigit = RunDigitInference(digitPatch);
                                    if (predictedDigit != null && lidToDigitColorMap.ContainsKey(lidColor))
                                    {
                                        string digitColor = lidToDigitColorMap[lidColor];
                                        resultText = $"{lidColor} 뚜껑\n{digitColor} 숫자 {predictedDigit}";

                                        // [신규] 중복집계 방지 - "지금 조합이 후보랑 같은가" 확인
                                        string detectionKey = $"{lidColor}_{digitColor}_{predictedDigit}";
                                        if (detectionKey == candidateKey)
                                        {
                                            candidateStableFrames++;
                                        }
                                        else
                                        {
                                            candidateKey = detectionKey;
                                            candidateStableFrames = 1;
                                        }

                                        // [카운트 확정] 몇 프레임 연속으로 안정적으로 같은 결과가 나왔고,
                                        // 그게 마지막으로 카운트했던 조합이랑 다르면 → 새 물체로 인정하고 카운트
                                        if (candidateStableFrames >= stabilityThreshold && detectionKey != lastCountedKey)
                                        {
                                            RecordInspection(lidColor, digitColor, predictedDigit);
                                            lastCountedKey = detectionKey;
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            // [신규] 뚜껑이 안 보이는 동안은 후보만 리셋 (손 지나감이든 진짜 치움이든 상관없음)
                            // lastCountedKey는 그대로 유지 - 손 지나갔다 같은 뚜껑으로 돌아오면
                            // 다시 candidateKey를 새로 쌓아야 하지만, lastCountedKey랑 같아서 재카운트는 안 됨
                            candidateKey = null;
                            candidateStableFrames = 0;
                        }

                        // [신규] Mat → Bitmap 변환은 Mat이 아직 살아있는(Dispose 되기 전) 이 시점에서 해야 함
                        System.Drawing.Bitmap mainBitmap = MatToBitmap(displayMat);
                        System.Drawing.Bitmap cropBitmap = MatToBitmap(croppedLid);

                        string statsTotalText = $"오늘 검사: {totalInspectedCount}건";
                        string statsColorText = $"화이트 {colorCounts["화이트"]}   파랑 {colorCounts["파랑"]}   빨강 {colorCounts["빨강"]}";
                        string statsDigitText = $"1번 {digitCounts["1"]}   2번 {digitCounts["2"]}   3번 {digitCounts["3"]}";

                        // [신규] 검사 기준(필터)이랑 비교해서 테두리 색 결정
                        // - 뚜껑이 없거나 기준이 "전체"면 중립 회색
                        // - 기준 색깔이랑 일치하면 초록(양품), 다르면 빨강(불량)
                        System.Drawing.Color borderColor;
                        if (lidColor == "없음" || expectedLidColor == null)
                        {
                            borderColor = System.Drawing.Color.FromArgb(70, 70, 70);
                        }
                        else if (lidColor == expectedLidColor)
                        {
                            borderColor = System.Drawing.Color.LimeGreen;
                        }
                        else
                        {
                            borderColor = System.Drawing.Color.Red;
                        }

                        // [신규] 실제 화면 갱신은 전부 UI 스레드에서 한 번에 처리
                        UpdateUi(() =>
                        {
                            var oldMain = pictureBox1.Image;
                            pictureBox1.Image = mainBitmap;
                            oldMain?.Dispose();

                            var oldCrop = pictureBoxCrop.Image;
                            pictureBoxCrop.Image = cropBitmap;
                            oldCrop?.Dispose();

                            // [신규] CNN 입력 미리보기 갱신 - digitPatchBitmap이 null이면
                            // (뚜껑이 없거나 숫자 추출 실패) 이전 이미지를 그대로 지워서 혼동 방지
                            var oldDigitPatch = pictureBoxDigitPatch.Image;
                            pictureBoxDigitPatch.Image = digitPatchBitmap;
                            oldDigitPatch?.Dispose();

                            lblResult.Text = resultText;
                            lblCaptureStatus.Text = autoCaptureStatusText;
                            lblTotalCount.Text = statsTotalText;
                            lblColorCounts.Text = statsColorText;
                            lblDigitCounts.Text = statsDigitText;
                            panelCameraBorder.BackColor = borderColor;
                        });
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
        // (배경 학습 마스크(GetForegroundMask) 기준으로 위치 찾음)
        private Mat CropLidArea(Mat colorMat)
        {
            using (Mat mask = GetForegroundMask(colorMat))
            {
                OpenCvSharp.Point[] biggest = GetBiggestCircularContour(mask);
                if (biggest == null)
                {
                    return colorMat.Clone();
                }

                Rect boundingBox = Cv2.BoundingRect(biggest);

                // [크래시 방지] 노이즈가 아주 얇은 조각으로 잘못 잡히면 뒤 단계(숫자추출)에서
                // AccessViolationException 나던 문제 있었음 - 너무 작으면 "못 찾음"과 동일하게 처리
                const int minCropSize = 20;
                if (boundingBox.Width < minCropSize || boundingBox.Height < minCropSize)
                {
                    return colorMat.Clone();
                }

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

            // [신규] 색 거리 계산 "전에" 먼저 블러 처리 - 글리터 반짝임 노이즈를 미리 주변과
            // 섞어서 완화함. 기존에 있던 블러(diffNormalized에 적용하는 것)는 반짝이는 픽셀이
            // 이미 극단적인 값을 만들어놓은 "뒤"라서 이미 늦은 시점이었음
            // ⚠️ 파이썬 train_digit_cnn.py의 extract_digit_patch()에도 반드시 똑같이 추가하고
            //    재학습해야 실제 효과가 있음 - C#만 바꾸면 학습 때와 다른 전처리를 넣는 셈이라 더 나빠짐
            Mat blurredForDiff = new Mat();
            Cv2.GaussianBlur(croppedLidBgr, blurredForDiff, new OpenCvSharp.Size(7, 7), 0);

            // [배경과의 색 거리 계산] 각 픽셀이 배경색이랑 얼마나 다른지(유클리드 거리)
            Mat floatMat = new Mat();
            blurredForDiff.ConvertTo(floatMat, MatType.CV_32FC3);
            blurredForDiff.Dispose();
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
        // [잠정치 - 아직 실측 전] 배경학습 마스크 기준(원형 뚜껑 자체 면적 대비)으로 계산 방식이 바뀌어서,
        // 이전 기준이 그대로 안 맞음 - 실제 로그 보고 다시 정확히 맞출 것
        private const double MinLidAreaRatioPercent = 3.0;
        private const double MaxLidAreaRatioPercent = 20.0;

        // (CropLidArea랑 똑같이 원형 체크(GetBiggestCircularContour)를 거친 뚜껑 모양만 보고 판정
        //  - 예전엔 화면 전체 전경 픽셀을 다 합쳐서 계산해서, 크롭엔 안 걸리는 자잘한 노이즈가
        //    판정에는 몰래 섞여 들어가는 문제가 있었음)
        private string ClassifyLidColor(Mat colorMat)
        {
            using (Mat fgMask = GetForegroundMask(colorMat))
            {
                OpenCvSharp.Point[] biggest = GetBiggestCircularContour(fgMask);
                if (biggest == null)
                {
                    return "없음";
                }

                // [신규] 원형 윤곽선 "안쪽만" 255로 채운 마스크를 새로 만듦 - 이 뚜껑 모양
                // 바깥에 남아있는 다른 노이즈 픽셀은 여기 안 들어와서 판정에 영향 안 줌
                using (Mat lidOnlyMask = Mat.Zeros(fgMask.Size(), MatType.CV_8UC1))
                using (Mat hsv = new Mat())
                {
                    Cv2.DrawContours(lidOnlyMask, new[] { biggest }, -1, Scalar.All(255), -1);

                    int maskPixels = Cv2.CountNonZero(lidOnlyMask);
                    double maskRatio = (double)maskPixels / (lidOnlyMask.Rows * lidOnlyMask.Cols) * 100;
                    System.Diagnostics.Debug.WriteLine($"뚜껑으로 인식된 영역: {maskRatio:F1}%");

                    // [뚜껑 없음 판정] 범위를 벗어나면(너무 작거나, 너무 크면) 뚜껑이 아닐 가능성이 높음
                    if (maskRatio < MinLidAreaRatioPercent || maskRatio > MaxLidAreaRatioPercent)
                    {
                        return "없음";
                    }

                    Cv2.CvtColor(colorMat, hsv, ColorConversionCodes.BGR2HSV);
                    Scalar meanHsv = Cv2.Mean(hsv, lidOnlyMask);
                    double avgHue = meanHsv.Val0;
                    double avgSat = meanHsv.Val1;
                    double avgVal = meanHsv.Val2;

                    System.Diagnostics.Debug.WriteLine(
                        $"평균 색상 - Hue:{avgHue:F1}, Sat:{avgSat:F1}, Val:{avgVal:F1}");

                    if (avgSat < 40)
                    {
                        return "화이트";
                    }
                    // [수정] 실측 결과 이 뚜껑의 빨강이 순수 빨강이 아니라 오렌지/구릿빛에 가까워서
                    // Hue가 40 근처로 나옴 - 기존 기준(25)보다 넉넉하게 45까지 확장
                    else if (avgHue <= 45 || avgHue >= 155)
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
        }

        // ch: 프로그램(창)을 닫을 때 실행되는 정리 함수
        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            isGrabbing = false;
            if (receiveThread != null)
            {
                receiveThread.Join();
            }

            // [신규] PictureBox에 남아있는 이미지 리소스 정리 (OpenCV 창은 이제 없어서 DestroyAllWindows 불필요)
            pictureBox1.Image?.Dispose();
            pictureBoxCrop.Image?.Dispose();

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
