from utils import LOGGER, colorstr



def print_samples(target, prediction):
    LOGGER.info('\n' + '-'*100)
    LOGGER.info(colorstr('Eval-Text: ') + target)
    LOGGER.info(colorstr('Predicted: ') + prediction)
    LOGGER.info('-'*100 + '\n')
