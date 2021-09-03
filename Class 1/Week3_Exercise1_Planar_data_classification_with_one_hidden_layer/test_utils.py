import numpy as np
from copy import deepcopy


def datatype_check(expected_output, target_output, error, level=0):
    success = 0
    if (level == 0):
        try:
            assert isinstance(target_output, type(expected_output))
            return 1
        except:
            return 0
    else:
        if isinstance(expected_output, tuple) or isinstance(expected_output, list) \
                or isinstance(expected_output, np.ndarray) or isinstance(expected_output, dict):
            if isinstance(expected_output, dict):
                range_values = expected_output.keys()
            else:
                range_values = range(len(expected_output))
            if len(expected_output) != len(target_output) or not isinstance(target_output, type(expected_output)):
                return 0
            for i in range_values:
                try:
                    success += datatype_check(expected_output[i],
                                            target_output[i], error, level - 1)
                except:
                    print("变量 {} 发生错误：{}, 目前类型: {}  期望类型{}".format(i,error,type(target_output[i]),type(expected_output[i])))
#                     print("Error: {} in variable {}, expected type: {}  but expected type{}".format(error,i,type(target_output[i]),type(expected_output[i])))
            if success == len(expected_output):
                return 1
            else:
                return 0

        else:
            try:
                assert isinstance(target_output, type(expected_output))
                return 1
            except:
                return 0


def equation_output_check(expected_output, target_output, error):
    success = 0
    if isinstance(expected_output, tuple) or isinstance(expected_output, list) or isinstance(expected_output, dict):
        if isinstance(expected_output, dict):
            range_values = expected_output.keys()
        else:
            range_values = range(len(expected_output))

        if len(expected_output) != len(target_output):
                return 0

        for i in range_values:
            try:
                success += equation_output_check(expected_output[i],
                                                 target_output[i], error)
            except:
                print("{}处变量发生错误：{}.".format(i, error))
#                 print("Error: {} for variable in position {}.".format(error, i))
        if success == len(expected_output):
            return 1
        else:
            return 0

    else:
        try:
            if hasattr(expected_output, 'shape'):
                np.testing.assert_array_almost_equal(
                    target_output, expected_output)
            else:
                assert target_output == expected_output
        except:
            return 0
        return 1


def shape_check(expected_output, target_output, error):
    success = 0
    if isinstance(expected_output, tuple) or isinstance(expected_output, list) or \
            isinstance(expected_output, dict) or isinstance(expected_output, np.ndarray):
        if isinstance(expected_output, dict):
            range_values = expected_output.keys()
        else:
            range_values = range(len(expected_output))

        if len(expected_output) != len(target_output):
                return 0
        for i in range_values:
            try:
                success += shape_check(expected_output[i],
                                       target_output[i], error)
            except:
                print("变量 {} 发生错误： {}.".format(i, error))
#                 print("Error: {} for variable {}.".format(error, i))
        if success == len(expected_output):
            return 1
        else:
            return 0

    else:
        return 1


def single_test(test_cases, target):
    success = 0
    for test_case in test_cases:
        try:
            if test_case['name'] == "datatype_check":
                assert isinstance(target(*test_case['input']),
                                  type(test_case["expected"]))
                success += 1
            if test_case['name'] == "equation_output_check":
                assert np.allclose(test_case["expected"],
                                   target(*test_case['input']))
                success += 1
            if test_case['name'] == "shape_check":
                assert test_case['expected'].shape == target(
                    *test_case['input']).shape
                success += 1
        except:
            print("错误: " + test_case['error'])

    if success == len(test_cases):
        print("\033[92m 测试全部通过")
    else:
        print('\033[92m', success, " 测试通过")
        print('\033[91m', len(test_cases) - success, " 测试失败")
        raise AssertionError(
            "{}部分测试没有通过. 请检查你的方程，避免在函数中使用全局变量".format(target.__name__))


def multiple_test(test_cases, target):
    success = 0
    for test_case in test_cases:
        try:
            test_input = deepcopy(test_case['input'])
            target_answer = target(*test_input)
        except:
            print('\33[30m', "错误，用这些输入运行测试案例时，解释器失败了: " + 
                  str(test_input))
            raise AssertionError("运行测试用例失败".format(target.__name__))

        try:
            if test_case['name'] == "datatype_check":
                success += datatype_check(test_case['expected'],
                                      target_answer, test_case['error'])
            if test_case['name'] == "equation_output_check":
                success += equation_output_check(
                    test_case['expected'], target_answer, test_case['error'])
            if test_case['name'] == "shape_check":
                success += shape_check(test_case['expected'],
                                   target_answer, test_case['error'])
        except:
            print('\33[30m', "错误: " + test_case['error'])

    if success == len(test_cases):
        print("\033[92m 测试全部通过")
    else:
        print('\033[92m', success, " 测试通过")
        print('\033[91m', len(test_cases) - success, " 测试失败")
        raise AssertionError(
            "{}部分测试没有通过. 请检查你的方程，避免在函数中使用全局变量".format(target.__name__))
