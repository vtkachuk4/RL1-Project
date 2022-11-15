import multiprocessing
from functools import partial
from main import main

if __name__ == "__main__":
    device = "cuda:1"
    activation = 'fta'
    use_target = True

    fta_params_dict = {
        # "fta_upper_limit": [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8],
        "fta_upper_limit": [20],
        "num_tiles": [20]
    }

    num_runs = 10
    pool = multiprocessing.Pool(processes=num_runs)
    
    for _fta_upper_limit in fta_params_dict["fta_upper_limit"]:
        _fta_lower_limit = -1*_fta_upper_limit
        for num_tiles in fta_params_dict["num_tiles"]:
            _fta_delta = (2*_fta_upper_limit)/num_tiles
            _fta_eta = _fta_delta
            
            print(
                "fta_upper_limit: %.2f  " % _fta_upper_limit,
                "fta_lower_limit: %.2f  " % _fta_lower_limit,
                "fta_delta: %.4f  " % _fta_delta,
                "fta_eta: %.4f  " % _fta_eta,
                # "run: ", run_i
                )

            runs = [i for i in range(num_runs)]
            main_i = partial(main, _use_target = use_target,
                                    activation = activation, 
                                    _fta_lower_limit = _fta_lower_limit, 
                                    _fta_upper_limit = _fta_upper_limit, 
                                    _fta_delta = _fta_delta, 
                                    _fta_eta = _fta_eta,
                                    _device = device)
            
            
            pool.map(main_i, runs)

            #     main(activation, _fta_lower_limit, _fta_upper_limit, _fta_delta, _fta_eta, run_i, device)
        #         break
        #     break
        # break