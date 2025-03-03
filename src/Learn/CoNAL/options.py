def model_opts(parser):
    parser.add_argument('--emb-dim', type=int, default=256)
    parser.add_argument('--hidden', default=[256, 128])
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--mode', default='train')

    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--new_data', type=bool, default=False)
    parser.add_argument('--num_samples', type=int, default=40000)
    parser.add_argument('--num_users', type=int, default=30)