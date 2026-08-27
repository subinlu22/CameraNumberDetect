using MvCameraControl;              // Hikrobot 카메라 SDK - 카메라 관련 모든 기능이 여기 들어있음
using System;
using System.Collections.Generic;   // 카메라 목록을 담을 List<> 쓰려고 필요함
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
            OpenCameraAndStartGrab();
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

            // [첫 번째 카메라 선택]
            IDeviceInfo deviceInfo = deviceInfoList[0];

            // [카메라 객체 생성]
            device = DeviceFactory.CreateDevice(deviceInfo);

            // [카메라 열기]
            result = device.Open();
            if (result != MvError.MV_OK)
            {
                MessageBox.Show("카메라 열기 실패! 에러코드: " + result);
                return;
            }

            // [GigE 패킷 크기 최적화]
            // 랜선(이더넷)으로 이미지 데이터를 주고받을 때, 카메라와 PC 네트워크 환경에
            // 딱 맞는 "한 번에 보낼 데이터 조각 크기"를 찾아서 설정하는 것
            // 이 설정이 없으면 이미지 데이터가 전송 중 깨져서 화면이 까맣게 나올 수 있음
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

            // [연속 촬영 모드 설정]
            device.Parameters.SetEnumValueByString("AcquisitionMode", "Continuous");
            device.Parameters.SetEnumValueByString("TriggerMode", "Off");

            // [영상 받기 시작]
            isGrabbing = true;

            // [카메라한테 "영상 흘려보내기 시작해" 명령] - 반드시 스레드 시작보다 먼저!
            device.StreamGrabber.StartGrabbing();

            // [별도 스레드 시작] - 프레임을 계속 받아오는 작업은 화면이 멈추지 않도록 따로 분리
            receiveThread = new Thread(ReceiveThreadProcess);
            receiveThread.Start();
        }

        // ch: 영상을 계속 반복해서 받아오는 함수 (별도 스레드에서 실행됨)
        private void ReceiveThreadProcess()
        {
            while (isGrabbing)
            {
                IFrameOut frameOut;

                // [프레임 한 장 받기] - 1000밀리초(1초) 동안 기다려보고 안 오면 포기
                int result = device.StreamGrabber.GetImageBuffer(1000, out frameOut);

                if (result == MvError.MV_OK)
                {
                    // [진단용] 실제로 받은 이미지 크기 확인
                    // 3072 x 2044 (카메라 실제 해상도)로 나오는지가 핵심 확인 포인트
                    System.Diagnostics.Debug.WriteLine(
                        "프레임 받음! 크기: " + frameOut.Image.Width + " x " + frameOut.Image.Height);

                    // [화면 갱신은 UI 스레드한테 부탁하기]
                    this.Invoke((MethodInvoker)delegate
                    {
                        try
                        {
                            device.ImageRender.DisplayOneFrame(pictureBox1.Handle, frameOut.Image);
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Debug.WriteLine("화면 그리기 실패: " + ex.Message);
                        }
                    });

                    // [메모리 반납] - 받은 프레임은 다 쓰고 나면 반드시 반납
                    device.StreamGrabber.FreeImageBuffer(frameOut);
                }
                else
                {
                    System.Diagnostics.Debug.WriteLine("프레임 받기 실패, 에러코드: " + result);
                }
            }
        }

        // ch: 프로그램(창)을 닫을 때 실행되는 정리 함수
        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            // [영상 받기 중단]
            isGrabbing = false;
            if (receiveThread != null)
            {
                receiveThread.Join();
            }

            // [카메라 정리] - StopGrabbing으로 "그만 흘려보내" 명령 (StartGrabbing 아님!)
            if (device != null)
            {
                device.StreamGrabber.StopGrabbing();
                device.Close();
                device.Dispose();
            }

            // [SDK 종료]
            SDKSystem.Finalize();
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }
    }
}