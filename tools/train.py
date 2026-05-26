# Copyright (c) CAIRI AI Lab. All rights reserved
import sys
import os.path
_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
_PROJ_ROOT = os.path.realpath(os.path.join(_CURR_DIR, '..'))
sys.path.append(os.path.join(_PROJ_ROOT))  # 添加openstl的包路径

# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           update_config)


if __name__ == '__main__':
    args = create_parser().parse_args()  # 接受解析后的参数
    config = args.__dict__  # 将参数转换为字典形式存储,config变化时,args.__dict__也会变化

    # 拼接所用方法的配置(参数)的路径
    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file

    # 是否允许重写配置
    if args.overwrite:  # 默认false
        config = update_config(config, load_config(cfg_path),
                               exclude_keys=['method'])
    else:
        loaded_cfg = load_config(cfg_path)  # 从配置路径中加载配置
        config = update_config(config, loaded_cfg,  # 将加载的配置信息合并到已经转换为字典存储的参数中
                               exclude_keys=['method', 'val_batch_size',
                                             'drop_path', 'warmup_epoch'])
        default_values = default_parser()  # 将最终配置的信息中为None的参数设置为默认值
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]

    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()