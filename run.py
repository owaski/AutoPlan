import os
import json
import time
import argparse
from functools import partial

from tqdm import tqdm
import wandb
from torch.utils.data import Dataset, DataLoader

from tasks import HotpotQA, ALFWorld

TASKS = {
    'hotpotqa': HotpotQA,
    'alfworld': ALFWorld
}


def evaluate(task, run_func, eval_start, eval_end, eval_set, eval_strategy=None, save_dir=None, verbose=False):
    os.makedirs(save_dir, exist_ok=True)
    
    if eval_strategy is None:
        eval_strategy = task.strategy

    n_correct = 0
    correct_mask = []

    if len(eval_set) == 0:
        eval_set = list(range(eval_start, eval_end + 1))

    for i in tqdm(eval_set):
        instance = task.data['dev'][i]
        success = False
        while not success:
            if task.price() > task.max_budget:
                print('Budget exceeded')
                break
            try:
                prediction, history = run_func([instance], strategy=eval_strategy, is_test=True, return_history=True, verbose=verbose)
            except Exception as e:
                print('Rerun question {} due to unexpected error: {}'.format(i, e))
                time.sleep(5)
                continue
            success = True
    
        score = task.score(instance, prediction[0])
        n_correct += int(score)
        correct_mask.append(score)

        print('Instance {}: \n\t{}\n\tPrediction: {}\n\tScore: {}\nCost: {}'.format(
            i, instance, prediction[0], score, task.price()
        ))

        if save_dir is not None:
            with open(os.path.join(save_dir, '{}.txt'.format(i)), 'w') as w:
                w.write(history)

        if task.price() > task.max_budget:
            break
    
    return {'n_correct': n_correct, 'correct_mask': correct_mask}


def train(args, task, run_func, n_train, n_epoch, batch_size, save_dir=None, verbose=False):
    # wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="meta-explore",
        entity="owaski",
        name="{}_bsz_{}_nepoch_{}".format(args.method, args.train_batch_size, args.n_epoch),
        # track hyperparameters and run metadata
        config=args.__dict__,
        mode="disabled" if 'debug' in args.method else "online",
    )

    os.makedirs(save_dir, exist_ok=True)

    strategies = [None]

    # with open(os.path.join(save_dir, 'init_strategy'), 'w') as w:
    #     w.write(strategies[0])

    dataloader = DataLoader(task.data['train'], batch_size=batch_size, 
                            shuffle=False, 
                            collate_fn=task.collate_fn)

    for epoch_idx in range(n_epoch):
        for i, instances in enumerate(tqdm(dataloader, desc='Epoch {}'.format(epoch_idx))):
            
            history_path = os.path.join(save_dir, '{}-{}-{}_history'.format(epoch_idx, i, batch_size))
            strategy_path = os.path.join(save_dir, '{}-{}-{}_strategy'.format(epoch_idx, i, batch_size))

            if os.path.exists(strategy_path):
                with open(strategy_path, 'r') as r:
                    strategies.append(r.read())
                print('[LOG] Loading saved strategy from {}'.format(strategy_path))
                continue

            success = False
            n_retry = 10
            while not success:
                if task.price() > task.max_budget:
                    print('Budget exceeded')
                    break
                try:
                    new_strategy, history = run_func(instances, strategy=strategies[-1],
                                                    return_history=True, verbose=verbose)
                except Exception as e:
                    print('Rerun question {} due to unexpected error: {}'.format(i, e))
                    time.sleep(10)
                    n_retry -= 1
                    if n_retry == 0:
                        break
                    continue
                success = True

            if not success:
                raise Exception('Retried this instance 10 times, quiting')

            with open(history_path, 'w') as w:
                w.write(history)

            with open(strategy_path, 'w') as w:
                w.write(new_strategy)
            
            strategies.append(new_strategy)

            print('Cost: {}'.format(task.price()))
            
            if task.price() > task.max_budget: 
                break

            n_train -= batch_size
            if n_train <= 0:
                break

    return strategies


def main(args):
    backend_args = {
        'name': args.backend,
        'access_token': args.access_token,
        'api_key': args.api_key,
        'org_id': args.org_id,
        'top_p': args.top_p,
        'temp': args.temp,
        'max_token': args.max_token,
        'presence_penalty': args.presence_penalty,
    }

    quota_args = {
        'sleep_minutes': args.sleep_minutes,
        'max_iter_per_instance': args.max_iter_per_instance,
        'max_budget': args.max_budget,
    }

    # add options for args.method
    task = TASKS[args.task](args.lm, args.method, backend_args, quota_args, subtask=args.subtask, train_bsz=args.train_batch_size, random_explore=args.random_explore)
    run_func = task.run
    if 'react' in args.method:
        run_func = partial(run_func, react=True)
    if 'retrieval' in args.method:
        run_func = partial(run_func, retrieval=True)

    save_tag = '{}_{}/{}_bsz_{}_nepoch_{}'.format(args.method, args.lm, args.subtask, args.train_batch_size, args.n_epoch)
    train_save_dir = os.path.join('results', args.task, save_tag, 'train')
    
    if args.train:
        train(args, task, run_func, args.n_train, args.n_epoch, 
              args.train_batch_size, train_save_dir, args.verbose)

    if args.eval:
        eval_strategy = None
        if args.eval_strategy is not None:
            with open(args.eval_strategy, 'r') as r:
                eval_strategy = r.read()
        
        eval_save_dir = os.path.join('results', args.task, save_tag, 'eval', 
                    os.path.basename(args.eval_strategy).split('.')[0] if args.eval_strategy else 'direct',
                    '{}_{}'.format(args.eval_start, args.eval_end))
        metrics = evaluate(task, run_func, args.eval_start, args.eval_end, args.eval_set, eval_strategy, eval_save_dir, args.verbose)

        # save metrics and history
        with open(os.path.join(eval_save_dir, 'eval.json'), 'w') as w:
            json.dump(metrics, w)
        


def parse_args():
    parser = argparse.ArgumentParser()

    # task and model
    parser.add_argument('--task', type=str, default='hotpotqa')
    parser.add_argument('--subtask', type=str, default='default') # only appliable to alfworld
    parser.add_argument('--lm', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--method', type=str, default='metaexplore')

    # quota
    parser.add_argument('--sleep-minutes', type=int, default=185)
    parser.add_argument('--max-iter-per-instance', type=int, default=20)
    parser.add_argument('--max-budget', type=float, default=10)

    # backend args
    parser.add_argument('--backend', type=str, default='revChatGPT')
    parser.add_argument('--access-token', type=str, default=None)
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--org-id', type=str, default=None)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--max-token', type=int, default=512)
    parser.add_argument('--presence-penalty', type=float, default=0.0)
    parser.add_argument('--verbose', action='store_true')

    # training and evaluation
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--n-train', type=int, default=5)
    parser.add_argument('--n-epoch', type=int, default=1)
    parser.add_argument('--train-batch-size', type=int, default=1)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval-strategy', type=str, default=None)
    parser.add_argument('--eval-set', type=int, nargs='+', default=[])
    parser.add_argument('--eval-start', type=int, default=0)
    parser.add_argument('--eval-end', type=int, default=0)

    # others
    parser.add_argument('--random-explore', type=float, default=0.0)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
