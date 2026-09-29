from openpi.history.observation_intervention import DualCameraBlackoutSchedule


def test_one_query_blackout_keeps_post_close_frame_in_recent_six() -> None:
    schedule = DualCameraBlackoutSchedule.from_first_close(
        55,
        replan_steps=5,
        blackout_queries=1,
    )

    assert schedule.first_post_close_query_step == 60
    assert schedule.clean_outcome_query_step == 65
    assert schedule.blackout_policy_query_steps == (70,)
    assert not schedule.is_blackout_observation(65)
    assert schedule.is_blackout_observation(66)
    assert schedule.is_blackout_observation(70)
    assert not schedule.is_blackout_observation(71)
    assert 64 in range(70 - 6, 70)


def test_two_query_blackout_marks_second_query_as_out_of_recent_six() -> None:
    schedule = DualCameraBlackoutSchedule.from_first_close(
        55,
        replan_steps=5,
        blackout_queries=2,
    )

    assert schedule.blackout_policy_query_steps == (70, 75)
    assert schedule.is_blackout_observation(75)
    assert not schedule.is_blackout_observation(76)
    assert 64 not in range(75 - 6, 75)


def test_mapping_round_trip() -> None:
    schedule = DualCameraBlackoutSchedule.from_first_close(
        30,
        replan_steps=5,
        blackout_queries=1,
    )

    assert DualCameraBlackoutSchedule.from_mapping(schedule.to_mapping()) == schedule
