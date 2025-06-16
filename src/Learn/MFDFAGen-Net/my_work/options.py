def model_opts(parser):
    parser.add_argument('--emb-dim',type = int,default = 256)
    parser.add_argument('--hidden',default = [256,128])
    parser.add_argument('--dropout',default = 0.5,type = float)
    parser.add_argument('--lr',default = 0.01,type = float)
    parser.add_argument('--num_epochs',default = 100,type = int)
    parser.add_argument('--mode',default = 'train')

    parser.add_argument('--batch_size',default = 512,type = int)


