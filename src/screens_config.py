from utils.screen_info import ScreenInfo

hardcoded_screens = [
    ScreenInfo(
        screen_id="1",
        screen_value=0,
        start_position = [[19, 144], [90, 128], [160, 80],  [260, 56]],
        first_frame=58,
        x_max=308, x_min=12,
        y_max=195, y_min=0,
        goal=[[ 250, 280], [0, 0]],
        next_screen_id = "2"
    ),
    ScreenInfo(
        screen_id="2",
        screen_value=1,
        start_position = [[264, -24], [360, -50], [403, -85], [445, -85], [530, -80]],
        first_frame=58,
        x_max=540, x_min=252,
        y_max=0, y_min=-190,
        goal=[[ 516, 540], [-190, -190]],
        next_screen_id = "3"
    ),
    ScreenInfo(
        screen_id="3",
        screen_value=2,
        start_position = [[528, -200], [645, -256], [580, -304], [600, -230], [508, -300], [700, -280], [760, -304]],
        first_frame=58,
        x_max=810, x_min=500,
        y_max=-170, y_min=-370,
        goal=[[ 764, 788], [-370, -370]],
        next_screen_id = "4"
    ),
    ScreenInfo(
        screen_id="4",
        screen_value=3,
        start_position = [[776, -392], [823, -480],[823, -480], [860, -475], [932, -480], [780, -450], [920, -520]],
        first_frame=58,
        x_max=1050, x_min=750,
        y_max=-360, y_min=-550,
        goal=[[ 908, 932], [-550, -550]],
        next_screen_id = "3b"
    ),
    ScreenInfo(
        screen_id="3b",
        screen_value=4,
        start_position = [[928, -568], [990, -584], [1110, -584], [1120, -672], [1035, -688],[1084, -680],[1028,-640]],
        first_frame=58,
        x_max=1180, x_min=880,
        y_max=-540, y_min=-735,
        goal=[[ 1075, 1052], [-735, -735]],
        next_screen_id = "5"
    ),
    ScreenInfo(
        screen_id="5",
        screen_value=5,
        start_position = [[1065, -752], [1140, -762], [1140, -880], [1230, -872], [1230, -872], [1320, -900]],
        first_frame=58,
        x_max=1340, x_min=1020,
        y_max=-700, y_min=-1020,
        goal=[[ 1310, 1325], [-1020, -1020]],
        next_screen_id = "6"
        ),
    ScreenInfo(
        screen_id="6",
        screen_value=6,
        start_position = [[1320, -1040], [1320, -1120], [1326, -1160], [1400, -1180], [1440, -1150], [1520, -1072], [1580, -1120]],
        first_frame=58,
        x_max=1620, x_min=1290,
        y_max=-1000, y_min=-1230,
        goal=[[ 1620, 1620], [-1128, -1220]],
        next_screen_id = "6a"
    ),
    ScreenInfo(
        screen_id="6a",
        screen_value=7,
        start_position = [[1624, -1128], [1643, -1080], [1700, -1040], [1732, -1125], [1833, -1136]],
        first_frame=58,
        x_max=2010, x_min=1620,
        y_max=-1000, y_min=-1230,
        goal=[[ 2010, 2010], [-1128, -1220]],
        next_screen_id = "6b"
    ),
    ScreenInfo(
        screen_id="6b",
        screen_value=8,
        start_position = [[2043, -1128], [2109, -1152], [2224, -1161], [2247, -1264]],
        first_frame=58,
        x_max=2323, x_min=2012,
        y_max=-1195, y_min=-1360,
        goal=[[ 2323, 2323], [-1304, -1363]],
        next_screen_id = "7"
    ),
    ScreenInfo(
        screen_id="6c",
        screen_value=9,
        start_position = [[2336, -1304], [2391,-1382],[2491,-1317],[2553,-1392],[2598,-1416]],
        first_frame=58,
        x_max=2612, x_min=2330,
        y_max=-1260, y_min=-1450,
        goal=[[ 2596, 2612], [-1450, -1450]],
        next_screen_id = "7"
    ),
    ScreenInfo(
        screen_id="7",
        screen_value=10,
        start_position = [[2609, -1480], [2706,-1515],[2727,-1595]],#[2272,-1576]
        first_frame=58,
        x_max=2812, x_min=2596,
        y_max=-1447, y_min=-1666,
        goal=[[ 2716, 2732], [-1666, -1666]],
        next_screen_id = "8"
    ),
    ScreenInfo(
        screen_id="8",
        screen_value=11,
        start_position = [[2728, -1704], [2762,-1792],[2915,-1736],[2991,-1808],[2907,-1792]],
        first_frame=58,
        x_max=3012, x_min=2708,
        y_max=-1680, y_min=-1829,
        goal=[[ 3012, 3012], [-1808, -1829]],
        next_screen_id = "8b"
    ),
    ScreenInfo(
        screen_id="8b",
        screen_value=12,
        start_position = [[3024,-1808],[3168,-1762],[3267,-1753],[3291,-1872]],
        first_frame=58,
        x_max=3224, x_min=3020,
        y_max=-1719, y_min=-1890,
        goal=[[ 3276, 3300], [-1890, -1890]],
        next_screen_id = "9"
    ),
    ScreenInfo(
        screen_id="9",
        screen_value=13,
        start_position = [[3286,-1928],[3444,-1993],[3435,-2048]],
        first_frame=58,
        x_max=3555, x_min=3275,
        y_max=-1928, y_min=-2068,
        goal=[[ 3555, 3555], [-2040, -2053]],
        next_screen_id = "9b"
    ),
    ScreenInfo(
        screen_id="9b",
        screen_value=14,
        start_position = [[3618,-2040],[3603,-2088],[3835,-2104],[3820,-2160]],
        first_frame=58,
        x_max=3874, x_min=3566,
        y_max=-2010, y_min=-2187,
        goal=[[ 3811, 3828], [-2187, -2187]],
        next_screen_id = "10a"
    ),
    ScreenInfo(
        screen_id="10a",
        screen_value=15,
        start_position = [[3848,-2216],[3925,-2317],[3966,-2336],[3839,-2344],[3748,-2239],[3729,-2320]],
        first_frame=58,
        x_max=3980, x_min=3708,
        y_max=-2410, y_min=-2187,
        goal=[[ 3708, 3724], [-2410, -2410]],
        next_screen_id = "11"
    ),
    ScreenInfo(
        screen_id="11",
        screen_value=16,
        start_position = [[3715,-2432],[3660,-2535],[3731,-2534],[3677,-2616],[3550,-2616],[3440,-2600]],
        first_frame=58,
        x_max=3748, x_min=3428,
        y_max=-2432, y_min=-2674,
        goal=[[ 3435, 3452], [-2674, -2674]],
        next_screen_id = "12"
    ),

    ScreenInfo(
        screen_id="12",
        screen_value=17,
        start_position = [[3440,-2696]],
        first_frame=58,
        x_max=3820, x_min=3436,
        y_max=-2696, y_min=-2893,
        goal=[[ 3820, 3820], [-2720, -2780]],
        next_screen_id = "10a"
    ),
    ScreenInfo(
        screen_id="12a",
        screen_value=18,
        start_position = [[3832,-2720]],
        first_frame=58,
        x_max=4132, x_min=3825,
        y_max=-2673, y_min=-3018,
        goal=[[ 3891, 3908], [-3018, -3018]],
        next_screen_id = "end"
    ),
    ScreenInfo(
        screen_id="end",
        screen_value=18,
        start_position = [[3896,-3048],[4007,-3080]],
        first_frame=58,
        x_max=4075, x_min=3844,
        y_max=-3048, y_min=-3154,
        goal=[[ 4075, 4075], [-3056, -3154]],
        next_screen_id = "nothing"
    ),
    ScreenInfo(
        screen_id="Bissy_00",
        screen_value=20,
        start_position = [[184,160], [253,48],[64,120],[152,63],[209,47]],
        first_frame=58,
        x_max=315, x_min=0,
        y_max=0, y_min=-200,
        goal=[[ 315, 315], [0, 48]],
        next_screen_id = "Bissy_01",
        map_id="SpringCollab2020/1-Beginner/Bissy"
    ),
    ScreenInfo(
        screen_id="a-00",
        screen_value=21,
        start_position = [[-255,136]],#, [-197,117],[-84,40],[-11,40],[78,-64]],
        first_frame=58,
        x_max=139, x_min=-323,
        y_max=-122, y_min=-179,
        goal=[[ 139, 139], [-122, -64]],
        next_screen_id = "a-01",
        map_id="4"
    ),
    # ScreenInfo(
    #     screen_id="a-01",
    #     screen_value=9,
    #     start_position = [[-303,144], [-197,117],[-84,40],[-11,40],[78,-64]],
    #     first_frame=58,
    #     x_max=139, x_min=-325,
    #     y_max=-122, y_min=-200,
    #     goal=[[ 139, 139], [-122, -64]],
    #     next_screen_id = "a-01x"
    # )
]