import argparse
import torch as th

def get_parser():
    parser = argparse.ArgumentParser(description='CCA-SSG')

    parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs.')
    parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate of CCA-SSG.')
    parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear evaluator.')
    parser.add_argument('--wd1', type=float, default=0, help='Weight decay of CCA-SSG.')
    parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay of linear evaluator.')

    parser.add_argument('--lambd', type=float, default=1e-3, help='trade-off ratio.')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')


    parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP instead of GNN')

    parser.add_argument('--der', type=float, default=0.2, help='Drop edge ratio.')
    parser.add_argument('--dfr', type=float, default=0.2, help='Drop feature ratio.')

    parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')
    parser.add_argument("--out_dim", type=int, default=512, help='Output layer dim.')

    # add by our self
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--debug', action='store_true', help='whether use the debug')
    parser.add_argument("--self_sup_type", type=str, default='cca', help='self_supervised methods.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and th.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'

        # jizhi config check
    import json, os

    WORKSPACE_PATH = os.environ.get('JIZHI_WORKSPACE_PATH', '')
    if WORKSPACE_PATH:
        args = vars(args)  # 转化为dict

        text = open(WORKSPACE_PATH + '/job_param.json', 'r').read()
        jizhi_json = json.loads(text)
        for key in jizhi_json:
            args[key] = jizhi_json[key]
        # Open Jizhi Reporter
        args['jizhi'] = True
        args = argparse.Namespace(**args)  # 转为namespace
    else:
        args.jizhi = False


    return args