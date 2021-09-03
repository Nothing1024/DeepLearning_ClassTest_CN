import numpy as np

         
def sigmoid_test(target):
    x = np.array([0, 2])
    output = target(x)
    assert type(output) == np.ndarray, "类型错误. 请使用 np.ndarray"
    assert np.allclose(output, [0.5, 0.88079708]), f"错误值. {output} != [0.5, 0.88079708]"
    output = target(1)
    assert np.allclose(output, 0.7310585), f"错误值. {output} != 0.7310585"
    print('\033[92m测试全部通过!')
    
            
        
def initialize_with_zeros_test(target):
    dim = 3
    w, b = target(dim)
    assert type(b) == float, f"b类型错误. {type(b)} != float"
    assert b == 0., "b 必须是 0.0"
    assert type(w) == np.ndarray, f"w类型错误. {type(w)} != np.ndarray"
    assert w.shape == (dim, 1), f"w类型错误. {w.shape} != {(dim, 1)}"
    assert np.allclose(w, [[0.], [0.], [0.]]), f"w值错误. {w} != {[[0.], [0.], [0.]]}"
    print('\033[92m测试全部通过!')

def propagate_test(target):
    w, b = np.array([[1.], [2.], [-1]]), 2.5, 
    X = np.array([[1., 2., -1., 0], [3., 4., -3.2, 1], [3., 4., -3.2, -3.5]])
    Y = np.array([[1, 1, 0, 0]])

    expected_dw = np.array([[-0.03909333], [ 0.12501464], [-0.99960809]])
    expected_db = np.float64(0.288106326429569)
    expected_grads = {'dw': expected_dw,
                      'db': expected_db}
    expected_cost = np.array(2.0424567983978403)
    expected_output = (expected_grads, expected_cost)
    
    grads, cost = target( w, b, X, Y)

    assert type(grads['dw']) == np.ndarray, f"梯度值['dw']错误. {type(grads['dw'])} != np.ndarray"
    assert grads['dw'].shape == w.shape, f"梯度值['dw']错误. {grads['dw'].shape} != {w.shape}"
    assert np.allclose(grads['dw'], expected_dw), f"梯度值['dw']错误. {grads['dw']} != {expected_dw}"
    assert np.allclose(grads['db'], expected_db), f"梯度值['dw']错误. {grads['db']} != {expected_db}"
    assert np.allclose(cost, expected_cost), f"代价函数值错误. {cost} != {expected_cost}"
    print('\033[92m测试全部通过!')

def optimize_test(target):
    w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
    expected_w = np.array([[-0.70916784], [-0.42390859]])
    expected_b = np.float64(2.26891346)
    expected_params = {"w": expected_w,
                       "b": expected_b}
   
    expected_dw = np.array([[0.06188603], [-0.01407361]])
    expected_db = np.float64(-0.04709353)
    expected_grads = {"dw": expected_dw,
                      "db": expected_db}
    
    expected_cost = [5.80154532, 0.31057104]
    expected_output = (expected_params, expected_grads, expected_cost)
    
    params, grads, costs = target(w, b, X, Y, num_iterations=101, learning_rate=0.1, print_cost=False)
    
    assert type(costs) == list, "costs类型错误. 应该是list类型"
    assert len(costs) == 2, f"costs长度错误. {len(costs)} != 2"
    assert np.allclose(costs, expected_cost), f"costs值错误. {costs} != {expected_cost}"
    
    assert type(grads['dw']) == np.ndarray, f"grads['dw']类型错误. {type(grads['dw'])} != np.ndarray"
    assert grads['dw'].shape == w.shape, f"grads['dw'] shape错误. {grads['dw'].shape} != {w.shape}"
    assert np.allclose(grads['dw'], expected_dw), f"grads['dw']值错误. {grads['dw']} != {expected_dw}"
    
    assert np.allclose(grads['db'], expected_db), f"grads['db']值错误. {grads['db']} != {expected_db}"
    
    assert type(params['w']) == np.ndarray, f"params['w']类型错误. {type(params['w'])} != np.ndarray"
    assert params['w'].shape == w.shape, f"params['w'] shape错误. {params['w'].shape} != {w.shape}"
    assert np.allclose(params['w'], expected_w), f"params['w']值错误. {params['w']} != {expected_w}"
    
    assert np.allclose(params['b'], expected_b), f"params['b']值错误. {params['b']} != {expected_b}"

    
    print('\033[92m测试全部通过!')   
        
def predict_test(target):
    w = np.array([[0.3], [0.5], [-0.2]])
    b = -0.33333
    X = np.array([[1., -0.3, 1.5],[2, 0, 1], [0, -1.5, 2]])
    
    pred = target(w, b, X)
    
    assert type(pred) == np.ndarray, f"pred类型错误. {type(pred)} != np.ndarray"
    assert pred.shape == (1, X.shape[1]), f"pred shape错误. {pred.shape} != {(1, X.shape[1])}"
    assert np.bitwise_not(np.allclose(pred, [[1., 1., 1]])), f"可能你忘算b了"
    assert np.allclose(pred, [[1., 0., 1]]), f"pred值错误. {pred} != {[[1., 0., 1.]]}"
    
    print('\033[92m测试全部通过!')
    
def model_test(target):
    np.random.seed(0)
    expected_output = {'costs': [np.array(0.69314718)],
                     'Y_prediction_test': np.array([[1., 1., 1., 1, 1, 1]]),
                     'Y_prediction_train': np.array([[1., 1., 1.]]),
                     'w': np.array([[ 0.00194946],
                            [-0.0005046 ],
                            [ 0.00083111],
                            [ 0.00143207]]),
                     'b': np.float64(0.000831188)
                      }
    
    # Use 3 examples for training
    dim, b, Y, X = 5, 3., np.array([1, 0, 1]).reshape(1, 3), np.random.randn(4, 3),

    # Use 6 examples for testing
    x_test = np.concatenate((X, X), axis=1)
    y_test = np.array([1, 0, 1, 1, 0, 1])
    
    d = target(X, Y, x_test, y_test, num_iterations=50, learning_rate=1e-4)
    
    assert type(d['costs']) == list, f"d['costs']类型错误. {type(d['costs'])} != list"
    assert len(d['costs']) == 1, f"d['costs']长度错误. {len(d['costs'])} != 1"
    assert np.allclose(d['costs'], expected_output['costs']), f"Wrong values for d['costs']. {d['costs']} != {expected_output['costs']}"
    
    assert type(d['w']) == np.ndarray, f"d['w']类型错误. {type(d['w'])} != np.ndarray"
    assert d['w'].shape == (X.shape[0], 1), f"d['w'] shape错误. {d['w'].shape} != {(X.shape[0], 1)}"
    assert np.allclose(d['w'], expected_output['w']), f"d['w']值错误. {d['w']} != {expected_output['w']}"
    
    assert np.allclose(d['b'], expected_output['b']), f"d['b']值错误. {d['b']} != {expected_output['b']}"
    
    assert type(d['Y_prediction_test']) == np.ndarray, f"d['Y_prediction_test']类型错误. {type(d['Y_prediction_test'])} != np.ndarray"
    assert d['Y_prediction_test'].shape == (1, x_test.shape[1]), f"d['Y_prediction_test'] shape错误. {d['Y_prediction_test'].shape} != {(1, X.shape[1])}"
    assert np.allclose(d['Y_prediction_test'], expected_output['Y_prediction_test']), f"d['Y_prediction_test']值错误. {d['Y_prediction_test']} != {expected_output['Y_prediction_test']}"
    
    assert type(d['Y_prediction_train']) == np.ndarray, f"d['Y_prediction_train']类型错误. {type(d['Y_prediction_train'])} != np.ndarray"
    assert d['Y_prediction_train'].shape == (1, X.shape[1]), f"d['Y_prediction_test'] shape错误. {d['Y_prediction_train'].shape} != {(1, X.shape[1])}"
    assert np.allclose(d['Y_prediction_train'], expected_output['Y_prediction_train']), f"d['Y_prediction_train']值错误. {d['Y_prediction_train']} != {expected_output['Y_prediction_train']}"
    
    print('\033[92m测试全部通过!')
    
