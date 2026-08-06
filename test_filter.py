# -*- coding: utf-8 -*-
"""ValueFilter 合理性校验回归测试（python test_filter.py）

覆盖：范围闸门 / 同回合递减 / 回合减小重置 / 行动值突变三帧确认 /
新对局回合增大三帧确认 / 开关关闭 / 候选污染与兜底重置。
说明：check() 成功后按真实监控流程调用 accept() 更新基线；
丢弃帧按真实流程调用 reject() 计数。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from starrail_monitor import ValueFilter  # noqa: E402

PASS = 0
FAIL = []


def ok(name, cond):
    global PASS
    if cond:
        PASS += 1
        print("PASS %s" % name)
    else:
        FAIL.append(name)
        print("FAIL %s" % name)


# 1. 首帧直接接受
def test_first_frame():
    vf = ValueFilter(max_turn=99, max_action=100)
    r = vf.check(1, 80)
    ok("首帧接受", r == (True, ""))
    vf.accept(1, 80)


# 2. 同回合递减接受
def test_decrease():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(1, 90)
    r1 = vf.check(1, 85)
    r2 = vf.check(1, 80)
    r3 = vf.check(1, 79)
    ok("同回合递减接受", r1 == (True, "") and r2 == (True, "")
       and r3 == (True, ""))
    vf.accept(1, 79)


# 3. 回合减小 = 合法重置
def test_turn_decrease_reset():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(1, 3)
    r = vf.check(0, 60)
    ok("回合减小重置接受", r == (True, ""))
    vf.accept(0, 60)


# 4. 突变开关关闭
def test_mutation_disabled():
    vf = ValueFilter(max_turn=99, max_action=100, allow_reset=False)
    vf.accept(1, 30)
    r = vf.check(1, 90)
    ok("突变开关关闭拒绝", r[0] is False and "开关关闭" in r[1])
    vf.reject()


# 5. 行动值突变三帧确认成功
def test_mutation_confirm():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(1, 30)
    r1 = vf.check(1, 90)
    r2 = vf.check(1, 90)
    r3 = vf.check(1, 90)
    ok("突变三帧确认", r1[0] is False and "待确认" in r1[1]
       and r2[0] is False and r3 == (True, "突变确认"))
    vf.accept(1, 90)


# 6. 突变三帧数值跳动 → 无效
def test_mutation_jitter():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(1, 30)
    r1 = vf.check(1, 90)
    r2 = vf.check(1, 95)
    r3 = vf.check(1, 92)
    ok("突变数值跳动无效", r1[0] is False and r2[0] is False
       and r3[0] is False and "无效" in r3[1])
    vf.reject()


# 7. 突变中间回落正常帧 → 候选清空
def test_mutation_interrupted():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(1, 30)
    r1 = vf.check(1, 90)   # 候选 1
    r2 = vf.check(1, 32)   # 回落正常帧，候选清空
    r3 = vf.check(1, 28)   # 正常递减
    ok("突变中间回落清空候选", r1[0] is False and r2 == (True, "")
       and r3 == (True, ""))
    vf.accept(1, 28)


# 8. 候选期间未识别帧（reject 计数）不影响候选
def test_mutation_ignore_frames():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(1, 30)
    r1 = vf.check(1, 90)   # 候选 1
    vf.reject()            # 未识别帧
    vf.reject()            # 未识别帧
    r2 = vf.check(1, 90)   # 候选 2
    r3 = vf.check(1, 88)   # 候选 3，递减 → 确认
    ok("未识别帧不干扰突变确认", r3 == (True, "突变确认"))
    vf.accept(1, 88)


# 9. 新对局：回合增大 3 帧等值 → 确认
def test_new_game_confirm():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(0, 4)
    r1 = vf.check(1, 100)
    r2 = vf.check(1, 100)
    r3 = vf.check(1, 98)
    ok("新对局回合增大确认", r1[0] is False and "回合增大" in r1[1]
       and r2[0] is False and r3 == (True, "新对局(回合增大确认)"))
    vf.accept(1, 98)


# 10. 新对局：单帧回合增大后回落 → 不误确认
def test_new_game_single_error():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(0, 4)
    r1 = vf.check(3, 90)   # 单帧误读
    r2 = vf.check(0, 4)    # 回落同值 → 正常接受，候选清空
    ok("单帧回合增大不误确认", r1[0] is False and r2 == (True, ""))
    vf.accept(0, 4)


# 11. 新对局：开关关闭 → 直接拒绝
def test_new_game_disabled():
    vf = ValueFilter(max_turn=99, max_action=100, allow_reset=False)
    vf.accept(0, 4)
    r = vf.check(1, 100)
    ok("新对局开关关闭拒绝", r[0] is False and "回合数增大" in r[1])
    vf.reject()


# 12. 突变不允许（回合=0 时不允许突变）
def test_mutation_low_action():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(0, 3)
    r = vf.check(0, 90)
    ok("回合0突变拒绝", r[0] is False and "突变不允许" in r[1])
    vf.reject()


# 13. 范围闸门
def test_range_gate():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(1, 30)
    r1 = vf.check(100, 30)
    r2 = vf.check(1, 101)
    ok("超范围拒绝", r1[0] is False and "超范围" in r1[1]
       and r2[0] is False and "超范围" in r2[1])
    vf.reject()


# 14. 连续丢弃 30 帧兜底重置基线
def test_drop_streak_reset():
    vf = ValueFilter(max_turn=99, max_action=100, reset_after=30)
    vf.accept(1, 30)
    for _ in range(29):
        vf.reject()
    r = vf.check(2, 90)    # 第 30 次 reject 前基线仍旧 → 增大拒绝
    ok("兜底前仍拒绝", r[0] is False)
    vf.reject()            # 第 30 次 → 基线重置
    r2 = vf.check(2, 90)   # last=None → 首帧接受
    ok("兜底后首帧接受", r2 == (True, ""))


# 15. 丢位防护：高位 99 被误读成 9 → 拒绝且不污染基线，后续 96 正常接受
def test_drop_digit_guard():
    vf = ValueFilter(max_turn=99, max_action=100, action_drop_max=20)
    vf.accept(0, 99)
    r1 = vf.check(0, 9)     # 丢位帧：99→9 骤降 90 → 拒绝
    r2 = vf.check(0, 96)    # 正确值：96 <= 99+5 → 正常接受
    ok("丢位帧拒绝且基线不污染", r1[0] is False and "骤降" in r1[1]
       and r2 == (True, ""))
    vf.accept(0, 96)


# 16. 用户场景：1回合0行动 → 0回合99 → 丢位9 → 96 仍应接受
def test_user_scenario():
    vf = ValueFilter(max_turn=99, max_action=100, action_drop_max=20)
    vf.accept(1, 0)
    r0 = vf.check(0, 99)    # 回合减小重置 → 接受
    ok("回合减小重置", r0 == (True, ""))
    vf.accept(0, 99)
    r1 = vf.check(0, 9)     # 丢位 → 拒绝
    r2 = vf.check(0, 96)    # 正确值 → 接受
    ok("丢位后正确值仍接受", r1[0] is False and r2 == (True, ""))


# 17. 骤降边界：降幅 20 允许，降幅 21 拒绝
def test_drop_boundary():
    vf = ValueFilter(max_turn=99, max_action=100, action_drop_max=20)
    vf.accept(0, 30)
    r1 = vf.check(0, 10)    # 降 20 → 允许
    ok("降幅20允许", r1 == (True, ""))
    vf.accept(0, 30)
    r2 = vf.check(0, 9)     # 降 21 → 拒绝
    ok("降幅21拒绝", r2[0] is False and "骤降" in r2[1])
    vf.reject()


# 18. 低位小降正常：不误伤正常递减
def test_drop_low_value():
    vf = ValueFilter(max_turn=99, max_action=100, action_drop_max=20)
    vf.accept(0, 10)
    r = vf.check(0, 8)
    ok("低位小降接受", r == (True, ""))


# 19. 位数骤降：两位数→一位数且降幅>5（丢十位）拒绝，即使降幅<20；
#    三位→两位（100→90 回合重置后的正常快速下降）不拦
def test_digit_drop():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(1, 15)
    r = vf.check(1, 3)
    ok("位数骤降拒绝(15→3)", r[0] is False and "骤降" in r[1])
    vf.reject()
    vf2 = ValueFilter(max_turn=99, max_action=100)
    vf2.accept(1, 100)
    r2 = vf2.check(1, 95)
    ok("三位→两位小幅下降接受(100→95)", r2 == (True, ""))
    vf2.accept(1, 95)
    vf4 = ValueFilter(max_turn=99, max_action=100)
    vf4.accept(0, 100)
    r4 = vf4.check(0, 90)
    ok("三位→两位快速下降接受(100→90)", r4 == (True, ""))
    vf4.accept(0, 90)
    vf3 = ValueFilter(max_turn=99, max_action=100)
    vf3.accept(1, 99)
    r3 = vf3.check(1, 9)
    ok("位数骤降拒绝(99→9)", r3[0] is False and "骤降" in r3[1])
    vf3.reject()


# 20. 未识别帧计数兜底：15 帧拒绝后基线重置，高位值可接受；重置时间戳更新
def test_reject_streak_recovery():
    vf = ValueFilter(max_turn=99, max_action=100, reset_after=15)
    vf.accept(0, 6)
    for _ in range(15):
        vf.reject()
    ok("兜底重置记录时间戳", vf.last is None and vf.reset_ts > 0)
    r = vf.check(0, 62)
    ok("兜底重置后高位值首帧接受", r == (True, ""))


# 21. 回合减小帧行动值低位拒绝（重置必为高位，低位=丢位误读）
def test_turn_drop_low_action():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(1, 1)
    r = vf.check(0, 9)
    ok("回合重置行动值低位拒绝(1→0, a9)", r[0] is False and "低位" in r[1])
    vf.reject()


# 22. 持续丢位三帧确认拒绝（真实99被读成9，3帧等值也不可信）
def test_persistent_drop_confirm_reject():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(1, 1)
    r1 = vf.check(1, 9)
    r2 = vf.check(1, 9)
    r3 = vf.check(1, 9)
    ok("突变确认值低位拒绝(3帧等值9)", r1[0] is False and r2[0] is False
       and r3[0] is False and "低位" in r3[1])
    vf.reject()


# 23. 新对局确认值低位拒绝（0→1 新对局但 action 9，持续丢位）
def test_new_game_low_action():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(0, 4)
    r1 = vf.check(1, 9)
    r2 = vf.check(1, 9)
    r3 = vf.check(1, 9)
    ok("新对局确认值低位拒绝", r1[0] is False and r2[0] is False
       and r3[0] is False and "低位" in r3[1])
    vf.reject()


# 24. 误报场景还原：基线(1,1)→突变确认低位拒→兜底重置→高位首帧接受
def test_false_alarm_scenario():
    vf = ValueFilter(max_turn=99, max_action=100, reset_after=15)
    vf.accept(1, 1)
    for _ in range(3):
        vf.check(1, 9)      # 突变候选3帧 → 确认值低位拒绝
        vf.reject()
    for _ in range(12):
        vf.reject()         # 兜底重置
    ok("兜底重置", vf.last is None)
    r = vf.check(0, 88)
    ok("重置后真实高位首帧接受", r == (True, ""))


# 25. 重置后首帧低位拒绝（16:14 误报场景：兜底重置后首帧 (0,4) 是丢位）
def test_reset_first_frame_low():
    vf = ValueFilter(max_turn=99, max_action=100, reset_after=15)
    vf.accept(1, 10)
    for _ in range(20):
        vf.reject()         # 连续未识别 → 兜底重置
    ok("兜底重置", vf.last is None)
    r = vf.check(0, 4)      # 重置后首帧低位 → 拒绝
    ok("重置后首帧低位拒绝(0,4)", r[0] is False and "首帧低位" in r[1])
    vf.reject()
    r2 = vf.check(0, 9)     # 仍低位 → 拒绝
    ok("重置后低位持续拒绝(0,9)", r2[0] is False and "首帧低位" in r2[1])
    vf.reject()
    r3 = vf.check(0, 88)    # 读到真实高位 → 接受
    ok("重置后真实高位接受(0,88)", r3 == (True, ""))


# 26. 启动首帧低位正常接受（reset_ts=0 不受冷却限制，防功能损失）
def test_startup_first_frame_low():
    vf = ValueFilter(max_turn=99, max_action=100)
    r = vf.check(0, 20)
    ok("启动首帧低位接受(0,20)", r == (True, ""))


# 27. 冷却过期后首帧低位接受（真实战斗尾声）
def test_reset_cooldown_expired():
    vf = ValueFilter(max_turn=99, max_action=100, reset_after=15)
    vf.accept(1, 10)
    for _ in range(15):
        vf.reject()
    ok("兜底重置", vf.last is None and vf.reset_ts > 0)
    vf.reset_ts -= 11      # 模拟冷却已过
    r = vf.check(0, 9)
    ok("冷却后首帧低位接受", r == (True, ""))


# 28. 归零回弹拒绝：行动值归零后同回合不回弹（新对局识别错值防护）
def test_zero_rebound():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(0, 0)
    r = vf.check(0, 1)
    ok("归零回弹拒绝(0,0)→(0,1)", r[0] is False and "回弹" in r[1])
    vf.reject()
    r2 = vf.check(0, 0)
    ok("归零保持接受(0,0)→(0,0)", r2 == (True, ""))
    vf.accept(0, 0)
    vf2 = ValueFilter(max_turn=99, max_action=100)
    vf2.accept(1, 0)
    r3 = vf2.check(1, 2)
    ok("1回合归零回弹拒绝(1,0)→(1,2)", r3[0] is False and "回弹" in r3[1])
    vf2.reject()
    r4 = vf2.check(0, 100)
    ok("归零后回合切换高位接受(1,0)→(0,100)", r4 == (True, ""))
    vf2.accept(0, 100)


# 29. 16:51 误报场景还原：战斗结束(0,0) → 新对局错值全拒 → 真实(1,89)新对局确认
def test_zero_rebound_scenario():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(0, 0)          # 16:49:20 战斗结束
    r1 = vf.check(0, 1)      # 16:51:04 新对局错值 (0,1) → 归零回弹拒
    r2 = vf.check(0, 9)      # 突变不允许拒
    r3 = vf.check(0, 92)     # 突变不允许拒
    r4 = vf.check(0, 4)      # 16:51:16 归零回弹拒（基线保持0）
    ok("战斗结束后的低位错值全拒", r1[0] is False and r2[0] is False
       and r3[0] is False and r4[0] is False)
    r5 = vf.check(1, 89)     # 真实新对局 (1,89) → 回合增大候选
    r6 = vf.check(1, 89)
    r7 = vf.check(1, 89)
    ok("真实新对局三帧确认", r7 == (True, "新对局(回合增大确认)"))
    vf.accept(1, 89)


# 30. 同位数骤降接受（游戏真实机制：86→54，两位数→两位数）
def test_same_digit_drop_accept():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(0, 86)
    r = vf.check(0, 54)
    ok("同位数骤降接受(86→54)", r == (True, ""))
    vf.accept(0, 54)
    r2 = vf.check(0, 49)
    ok("骤降后继续递减接受(54→49)", r2 == (True, ""))
    vf.accept(0, 49)
    # 误读自愈：86→56 误读接受后，真实 54 容差内接受
    vf2 = ValueFilter(max_turn=99, max_action=100)
    vf2.accept(0, 86)
    r3 = vf2.check(0, 56)
    ok("同位数骤降误读接受(86→56)", r3 == (True, ""))
    vf2.accept(0, 56)
    r4 = vf2.check(0, 54)
    ok("误读后真实值容差自愈(56→54)", r4 == (True, ""))


# 31. 多位数→个位数丢位拒绝（99→9 / 100→9 / 86→8）
def test_digit_drop_to_unit():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(0, 86)
    r = vf.check(0, 8)
    ok("丢位拒绝(86→8)", r[0] is False and "骤降" in r[1])
    vf.reject()
    vf2 = ValueFilter(max_turn=99, max_action=100)
    vf2.accept(0, 100)
    r2 = vf2.check(0, 9)
    ok("丢位拒绝(100→9)", r2[0] is False and "骤降" in r2[1])
    vf2.reject()
    vf3 = ValueFilter(max_turn=99, max_action=100)
    vf3.accept(0, 99)
    r3 = vf3.check(0, 9)
    ok("丢位拒绝(99→9)", r3[0] is False and "骤降" in r3[1])
    vf3.reject()


# 32. 今日场景还原：0回合86 → 真实骤降54/49/48/46 全接受
def test_today_scenario():
    vf = ValueFilter(max_turn=99, max_action=100)
    vf.accept(0, 86)
    ok1 = vf.check(0, 54) == (True, "")
    vf.accept(0, 54)
    ok2 = vf.check(0, 49) == (True, "")
    vf.accept(0, 49)
    ok3 = vf.check(0, 48) == (True, "")
    vf.accept(0, 48)
    ok4 = vf.check(0, 46) == (True, "")
    vf.accept(0, 46)
    ok("今日场景：86→54→49→48→46 全部接受", ok1 and ok2 and ok3 and ok4)


def main():
    test_first_frame()
    test_decrease()
    test_turn_decrease_reset()
    test_mutation_disabled()
    test_mutation_confirm()
    test_mutation_jitter()
    test_mutation_interrupted()
    test_mutation_ignore_frames()
    test_new_game_confirm()
    test_new_game_single_error()
    test_new_game_disabled()
    test_mutation_low_action()
    test_range_gate()
    test_drop_streak_reset()
    test_drop_digit_guard()
    test_user_scenario()
    test_drop_boundary()
    test_drop_low_value()
    test_digit_drop()
    test_reject_streak_recovery()
    test_turn_drop_low_action()
    test_persistent_drop_confirm_reject()
    test_new_game_low_action()
    test_false_alarm_scenario()
    test_reset_first_frame_low()
    test_startup_first_frame_low()
    test_reset_cooldown_expired()
    test_zero_rebound()
    test_zero_rebound_scenario()
    test_same_digit_drop_accept()
    test_digit_drop_to_unit()
    test_today_scenario()
    print("----")
    print("%d 项通过" % PASS)
    if FAIL:
        print("失败: %s" % ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
