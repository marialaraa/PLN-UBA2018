"""Print corpus statistics.

Usage:
  stats.py
  stats.py -h | --help

Options:
  -h --help     Show this screen.
"""
from docopt import docopt

from sentiment.tass import InterTASSReader, GeneralTASSReader


def reader_statistic(reader_name, reader):
    polarity = list(reader.y())
    print('Estadísticas de %s' % reader_name)
    print('================')
    print('Cantidad total de tweets: {}'.format(len(list(reader.tweets()))))
    print('Cantidad de tweets con polaridad P: {}'.format(polarity.count('P')))
    print('Cantidad de tweets con polaridad N: {}'.format(polarity.count('N')))
    print('Cantidad de tweets con polaridad NEG: {}'.format(polarity.count('NEU')))
    print('Cantidad de tweets con polaridad NONE: {}'.format(polarity.count('NONE')))
    print('================\n\n')


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    inter_tass_reader = InterTASSReader('../corpus/TASS/InterTASS/tw_faces4tassTrain1000rc.xml')
    general_tass_reader = GeneralTASSReader('../corpus/TASS/GeneralTASS/general-tweets-train-tagged.xml',
                                            simple=True)

    reader_statistic('InterTass', inter_tass_reader)
    reader_statistic('GeneralTass', general_tass_reader)
