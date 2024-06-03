from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

hand_static = pipeline(Tasks.hand_static, model='damo/cv_mobileface_hand-static')
result_status = hand_static("C:\\Users\\86151\\Desktop\\face\\yeye.jpg")
result = result_status[OutputKeys.OUTPUT]

dir = {
"bixin":"单手比心",

"d_bixin":"双手比心",

"d_first_left":"左手抱拳",

"d_fist_right":"右手抱拳",

"d_hand":"双手交叉",

"fashe":"发射形状",

"fist":"握拳",

"five":"手掌张开",

"ok":"ok",

"one":"用手比数字1",

"tuoju":"托举手势",

"two":"比耶",

"yaogun":"摇滚手势",

"zan":"点赞",

"unrecog":"未识别"
}
result_val = dir.get(result, '未识别')
print(result_val)
