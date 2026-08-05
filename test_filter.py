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
    r2 = vf.check(0, 3)    # 回落，回合减小 → 接受
    ok("单帧回合增大不误确认", r1[0] is False and r2 == (True, ""))
    vf.accept(0, 3)


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
    print("----")
    print("%d 项通过" % PASS)
    if FAIL:
        print("失败: %s" % ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
