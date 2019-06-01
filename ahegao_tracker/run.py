from ahegao import Ahegao
from helper import *


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args = parse_args()
    ahegao=Ahegao(args)
    ahegao.run()
