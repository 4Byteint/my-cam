class GripperState:
    """state"""
    STATE_POWER_OFF = 1
    STATE_POWER_ON = 2
    STATE_INITIALIZING = 3
    STATE_RELEASING = 4
    STATE_GRABBING = 5
    STATE_OFFLINE = 6
    STATE_GRABBING_MISS = 7
    STATE_RELEASING_MISS = 8

    # only for readable printing
    stateDict = {
        STATE_POWER_ON: "state Power On",
        STATE_POWER_OFF: "state Power Off",
        STATE_INITIALIZING: "state Initialized",
        STATE_RELEASING: "state Releasing",
        STATE_GRABBING: "state Holding",
        STATE_OFFLINE: "state Offline",
        STATE_GRABBING_MISS: "state Grabbing Miss",
        STATE_RELEASING_MISS: "state Releasing Miss",
    }


class ArmCmd:
    """cmd"""
    # state switching cmd
    CMD_ERROR = -1
    CMD_NO_NEWCMD = 0
    CMD_RELEASE = 1  #:
    CMD_GRAB = 2
    CMD_INIT = 3  #:初始化, 各裝置接初始化狀態(等待夾取相關命令到來前皆進入初始化),夾爪到初始位置,目前位置與放開狀態相同
    CMD_POWERON = 4  #: 重開機, 包含開機確認,各裝置初始化確認
    CMD_POWEROFF = 5

    # state remaining cmd
    CMD_STATE_CHECK = 6

    # only for readable printing
    cmdDict = {
        CMD_POWERON: "cmd Power On",
        CMD_POWEROFF: "cmd Power Off",
        CMD_INIT: "cmd Initialization",
        CMD_RELEASE: "cmd Releas",
        CMD_GRAB: "cmd Hold",
        CMD_STATE_CHECK: "cmd State Check",
        CMD_NO_NEWCMD: "no New Cmd",
    }


class GripperInfomation:
    NO_INFO = 0
    GRIPPER_START_GRABBING = 1
    GRIPPER_START_RELEASING = 2
    GRIPPER_INITIAL_OK = 3
    GRIPPER_INITIAL_NOTOK = 4
    GRIPPER_OFFLINE = 5
    GRABBING_MISS = 6
    RELEASING_MISS = 7


class CanData:

    # (0, instruction, from whom , index)
    # instruction :  --- 1 : CMD --- 2 : DATA --- 3 : STATE
    # from whom : --- 1 : PI --- 2 : STM --- 3 : UNO

    CAN_ERROR_FRAME = (0, 0, 0, 0)

    CMD_PI_CONNECTION_CHECK = (0, 1, 1, 1)
    CMD_PI_UNO_INIT = (0, 1, 1, 2)
    CMD_PI_STM_INIT = (0, 1, 1, 3)
    CMD_PI_GRABBING = (0, 1, 1, 4)
    CMD_PI_RELEASING = (0, 1, 1, 5)
    CMD_PI_SENSOR_REQUEST = (0, 1, 1, 6)

    STATE_UNO_CONNECTCHECK = (0, 3, 3, 1)
    STATE_UNO_INIT_OK = (0, 3, 3, 2)
    STATE_UNO_INIT_NOTOK = (0, 3, 3, 3)

    STATE_STM_CONNECTCHECK = (0, 3, 2, 1)
    STATE_STM_INIT_OK = (0, 3, 2, 2)
    STATE_STM_INIT_NOTOK = (0, 3, 2, 3)
    STATE_STM_START_GRABBING = (0, 3, 2, 4)
    STATE_STM_START_RELEASING = (0, 3, 2, 5)
    STATE_STM_MOTOR_OFFLINE = (0, 3, 2, 6)
    STATE_STM_GRABBING_MISS = (0, 3, 2, 7)
    STATE_STM_RELEASING_MISS = (0, 3, 2, 8)

    DATA_UNO_SENSOR_DATA = (0, 2, 3, 1)


class CanId:
    CANID_PI_TO_ALL = 1
    CANID_PI_TO_STM = 2
    CANID_PI_TO_UNO = 3
    


class Device:
    STM = 1
    UNO = 2


class Status:
    SUCCESS = 1
    FAILED = -1
    UNKNOWN = 0