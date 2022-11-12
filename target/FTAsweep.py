from main import main

if __name__ == "__main__":
    activation = 'fta'
    fta_params_dict = {
        "fta_upper_limit": [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8],
        "num_tiles": [14, 16, 18, 20, 21, 22]
    }
    num_runs = 10

    for _fta_upper_limit in fta_params_dict["fta_upper_limit"]:
        _fta_lower_limit = -1*_fta_upper_limit
        for num_tiles in fta_params_dict["num_tiles"]:
            _fta_delta = (2*_fta_upper_limit)/num_tiles
            _fta_eta = _fta_delta
            
            for run_i in range(num_runs):
                print(
                "fta_upper_limit: %.2f  " % _fta_upper_limit,
                "fta_lower_limit: %.2f  " % _fta_lower_limit,
                "fta_delta: %.4f  " % _fta_delta,
                "fta_eta: %.4f  " % _fta_eta,
                "run: ", run_i
                )
                main(activation, _fta_lower_limit, _fta_upper_limit, _fta_delta, _fta_eta, run_i)
                break
            break
        break