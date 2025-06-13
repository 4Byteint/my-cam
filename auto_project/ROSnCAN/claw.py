# from canlib import canlib, Frame
import time
from can_table import (
    GripperState,
    ArmCmd,
    GripperInfomation,
    CanData,
    CanId,
    Device,
    Status,
)
import os
import can
from can_sender import CANInterface

class Claw:
    """This class handles all CAN-related tasks using python-can."""
    def __init__(self, channel='can0', bitrate=1000000):
        """ init CAN channel and set up initial state """
        self.can = CANInterface(channel=channel, bitrate=bitrate)
        # --------------------- package in can_module --------------------- #
        # self.ch = canlib.openChannel(
        #     channel=0, flags=canlib.Open.EXCLUSIVE, bitrate=canlib.canBITRATE_1M
        # )
        # self.ch.setBusOutputControl(canlib.Driver.NORMAL)
        # self.ch.busOn()
        # self.canMsg = Frame(id_=0, data=[0, 0, 0, 0, 0, 0, 0, 0], dlc=8)
        # --------------------- package in can_module --------------------- #
        
        self.connectCheckTimeStart = 0
        self.sendingTimeStart_STM = 0
        self.sendingTimeStart_UNO = 0

        self.connectCheckPrintFlag = True

        # public
        self.canData = (0, 0, 0, 0, 0, 0, 0, 0)
        self.state = GripperState.STATE_POWER_OFF  # 初始狀態

        #  True for CAN  been transmitted for the first time
        self.sendFirstTimeFlag = {Device.STM: False, Device.UNO: False}
        self.connectStatus = {Device.STM: Status.UNKNOWN, Device.UNO: Status.UNKNOWN}
        self.initStatus = {Device.STM: Status.UNKNOWN, Device.UNO: Status.UNKNOWN}

        # state task to do
        self.toDoTask = {
            GripperState.STATE_POWER_OFF: self.PowerOff,
            GripperState.STATE_POWER_ON: self.PowerOn,
            GripperState.STATE_INITIALIZING: self.Initialization,
            GripperState.STATE_RELEASING: self.Release,
            GripperState.STATE_GRABBING: self.Grab,
            GripperState.STATE_OFFLINE: self.OffLine,
            GripperState.STATE_GRABBING_MISS: self.GrabbingMiss,
            GripperState.STATE_RELEASING_MISS: self.ReleasingMiss,
        }

    # ********************************************************************* #
    # ********************************************************************* #
    # ************************ Can Reading ******************************** #
    # ********************************************************************* #
    # ********************************************************************* #
    def readCanBlocking(self):
        """a blocking CAN data reading method"""
        try:
            # this line is a fucking blocking method
            self.canMsg = self.ch.read(timeout=10)

            self.canData = tuple(self.canMsg.data)
            print(self.canMsg)
        except canlib.canNoMsg:
            self.canData = CanData.CAN_NO_MSG + (0, 0, 0, 0)
        except canlib.canError:
            self.canData = CanData.CAN_ERROR + (0, 0, 0, 0)

    # ********************************************************************* #
    # ********************************************************************* #
    # *************** Connection Check Related Func.*********************** #
    # ********************************************************************* #
    # ********************************************************************* #

    def ConnectCheck(self):

        timePass = time.time() - self.connectCheckTimeStart

        # do the connect check every 10sec
        if timePass > 10:
            print("connection check sending")
            self.connectCheckTimeStart = time.time()
            self.connectCheckPrintFlag = True
            self.connectStatus[Device.UNO] = Status.UNKNOWN
            self.connectStatus[Device.STM] = Status.UNKNOWN
            self.ch.write(
                Frame(
                    id_=CanId.CANID_PI_TO_ALL,
                    data=list(CanData.CMD_PI_CONNECTION_CHECK) + [0, 0, 0, 0],
                    dlc=8,
                )
            )
        # reconnecting every 1 sec if one of devices is still unchecked
        elif timePass > 1:
            if (
                self.connectStatus[Device.STM] == Status.UNKNOWN
                or self.connectStatus[Device.UNO] == Status.UNKNOWN
            ):
                self.connectCheckTimeStart = time.time()
                self.connectCheckPrintFlag = True
                print("connection resend")
                self.ch.write(
                    Frame(
                        id_=CanId.CANID_PI_TO_ALL,
                        data=list(CanData.CMD_PI_CONNECTION_CHECK) + [0, 0, 0, 0],
                        dlc=8,
                    )
                )

        if self.connectCheckPrintFlag:
            # print("connect status")
            # print(self.connectStatus[Device.STM])
            # print(self.connectStatus[Device.UNO])
            if (
                self.connectStatus[Device.STM] == Status.SUCCESS
                and self.connectStatus[Device.UNO] == Status.SUCCESS
            ):
                print("connection success")
            else:
                print("connection not success")
            self.connectCheckPrintFlag = False

    def ConnectStatusUpdate(self):
        """called before self.canData cleared or covered"""

        if self.canData[0:4] == CanData.STATE_UNO_CONNECTCHECK:
            self.connectStatus[Device.UNO] = Status.SUCCESS
            print("UNO_CONNNECT_SUCCESS")
        elif self.canData[0:4] == CanData.STATE_STM_CONNECTCHECK:
            self.connectStatus[Device.STM] = Status.SUCCESS
            print("STM_CONNNECT_SUCCESS")

    # ********************************************************************* #
    # ********************************************************************* #
    # ********************* State Machine Task **************************** #
    # ********************************************************************* #
    # Every state has its own state-task todo, task return class "Status"   #
    # when it's done.   *************************************************** #
    # ********************************************************************* #

    def PowerOff(self):
        """tell STM to turn off motor"""
        # print("power off")

        return Status.SUCCESS

    def PowerOn(self):
        """"""
        # print("power on")

        # 上電後的初始化操作，例如檢查電源、啟動系統、檢查各個device是否開啟
        # while True:
        #   # 在背景執行 process
        #   process = subprocess.Popen(["python3", "digit_check.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #   time.sleep(3)
        #   if process.poll() is None:
        #     print("camera is running.")
        #     self.state = "CheckConnection"
        #     break
        #   else:
        #     print("digit not found")

        # wait for connection check
        if (
            self.connectStatus[Device.UNO] == Status.SUCCESS
            and self.connectStatus[Device.STM] == Status.SUCCESS
        ):
            # **************************************************************************************
            # automatically go into init state
            self.state = GripperState.STATE_INITIALIZING
            self.sendFirstTimeFlag[Device.UNO] = True
            self.sendFirstTimeFlag[Device.STM] = True
            return Status.SUCCESS

    def Initialization(self):
        # print("init")

        if self.sendFirstTimeFlag[Device.UNO]:
            print("init UNO send")
            self.ch.write(
                Frame(
                    id_=CanId.CANID_PI_TO_UNO,
                    data=list(CanData.CMD_PI_UNO_INIT) + [0, 0, 0, 0],
                    dlc=8,
                )
            )
            self.sendFirstTimeFlag[Device.UNO] = False
            self.sendingTimeStart_UNO = time.time()
        else:
            received = self.canData[0:4]
            if received == CanData.STATE_UNO_INIT_OK:
                print("UNO INIT SUCCCESS")
                self.initStatus[Device.UNO] = Status.SUCCESS
            elif received == CanData.STATE_UNO_INIT_NOTOK:
                print("UNO INIT FAILED")
                self.initStatus[Device.UNO] = Status.FAILED
            elif time.time() - self.sendingTimeStart_UNO > 1500:
                self.sendFirstTimeFlag[Device.UNO] = True
                self.initStatus[Device.UNO] = Status.FAILED
                print("未收到UNO INIT,重新發送")

        if self.sendFirstTimeFlag[Device.STM]:
            print("init STM send")
            self.ch.write(
                Frame(
                    id_=CanId.CANID_PI_TO_STM,
                    data=list(CanData.CMD_PI_STM_INIT) + [10, 0, 0, 0],
                    dlc=8,
                )
            )
            self.sendFirstTimeFlag[Device.STM] = False
            self.sendingTimeStart_STM = time.time()
        else:
            received = self.canData[0:4]
            if received == CanData.STATE_STM_INIT_OK:
                print("STM INIT SUCCCESS")
                self.initStatus[Device.STM] = Status.SUCCESS
            elif received == CanData.STATE_STM_INIT_NOTOK:
                print("STM INIT FAILED")
                self.initStatus[Device.STM] = Status.FAILED
            elif time.time() - self.sendingTimeStart_STM > 1500:
                self.sendFirstTimeFlag[Device.STM] = True
                self.initStatus[Device.STM] = Status.FAILED
                print("未收到STM INIT,重新發送")

        if (
            self.initStatus[Device.STM] == Status.SUCCESS
            and self.initStatus[Device.UNO] == Status.SUCCESS
        ):
            return Status.SUCCESS
        elif (
            self.initStatus[Device.STM] == Status.FAILED
            or self.initStatus[Device.UNO] == Status.FAILED
        ):
            return Status.FAILED
        else:
            return Status.UNKNOWN

    def Grab(self):

        if self.sendFirstTimeFlag[Device.UNO]:
            self.ch.write(
                Frame(
                    id_=CanId.CANID_PI_TO_UNO,
                    data=list(CanData.CMD_PI_SENSOR_REQUEST) + [0, 0, 0, 0],
                    dlc=8,
                )
            )
            self.sendFirstTimeFlag[Device.UNO] = False

        #  remove if and keep else if you want to send STM cmd Constantly
        if self.sendFirstTimeFlag[Device.STM]:

            # ******************************************
            # self.sendingTimeStart_STM = time.time()
            # *******************************************

            self.ch.write(
                Frame(
                    id_=CanId.CANID_PI_TO_STM,
                    data=list(CanData.CMD_PI_GRABBING) + [200, 0, 0, 0],
                    dlc=8,
                )
            )
            self.sendFirstTimeFlag[Device.STM] = False

            # ******************************************
            # self.sendingTimeStart_STM = time.time()
            # ******************************************

        else:
            # try:

            if self.canData[0:4] == CanData.STATE_STM_START_GRABBING:

                # *************************************************
                # canTime = time.time() - self.sendingTimeStart_STM
                # with open("CanToCanDelay.txt", "a") as file:
                #     file.write(f"{canTime}\n")
                # *************************************************

                print("start grabbing success")
                return Status.SUCCESS

            # except canlib.CanNoMsg:
            #     # if time.time() - self.sendingTimeStart_STM > 5:
            #     self.sendFirstTimeFlag[Device.STM] = True
            #     print("未收到STM是否開夾,重新發送")
            #     return Status.UNKNOWN

    def Release(self):

        if self.sendFirstTimeFlag[Device.STM]:
            self.ch.write(
                Frame(
                    id_=CanId.CANID_PI_TO_STM,
                    data=list(CanData.CMD_PI_RELEASING) + [10, 0, 0, 0],
                    dlc=8,
                )
            )
            self.sendFirstTimeFlag[Device.STM] = False
            # self.sendingTimeStart_STM = time.time()
            # print(time.time() - self.sendingTimeStart_STM)
        else:
            # try:

            if self.canData[0:4] == CanData.STATE_STM_START_RELEASING:
                return Status.SUCCESS

            # except canlib.CanNoMsg:
            #     # if time.time() - self.sendingTimeStart_STM > 5:
            #     self.sendFirstTimeFlag[Device.STM] = True
            #     print("未收到STM是否放開,重新發送")
            #     return Status.UNKNOWN

    def OffLine(self):
        """nothing to do with STM,UNO"""
        # print("offline")
        return Status.SUCCESS

    def GrabbingMiss(self):
        """nothing to do with STM,UNO"""
        # print("grabing miss")
        return Status.SUCCESS

    def ReleasingMiss(self):
        """nothing to do with STM,UNO"""
        # print("releasing miss")
        return Status.SUCCESS

    # ********************************************************************* #
    # ********************************************************************* #
    # ************************* Other Task Func. ************************** #
    # ********************************************************************* #
    # ********************************************************************* #
    def NoTask(self):
        """nothing to do with STM,UNO"""
        print("no task")
        return Status.SUCCESS

    def shutdown(self):
        # 關閉機器
        self.ch.busOff()
        self.ch.close()