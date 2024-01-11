import textgrid

class Segment(object):
    def __init__(self, uttid, spkr, stime, etime, text):
        self.uttid = uttid
        self.spkr = spkr
        self.stime = round(stime, 2)
        self.etime = round(etime, 2)
        self.text = text

    def change_stime(self, time):
        self.stime = time

    def change_etime(self, time):
        self.etime = time


def convert_textgrid_rttm(file):
    tg = textgrid.TextGrid.fromFile(file)
    segments = []
    spk = {}
    num_spk = 1
    for i in range(tg.__len__()):
        for j in range(tg[i].__len__()):
            if tg[i][j].mark:
                if tg[i].name not in spk:
                    spk[tg[i].name] = num_spk
                    num_spk += 1
                segments.append(
                    Segment(
                        spk[tg[i].name],
                        tg[i][j].minTime,
                        tg[i][j].maxTime,
                        tg[i][j].mark.strip(),
                    )
                )
    segments = sorted(segments, key=lambda x: x.stime)

    rttm = ""

    for i in range(len(segments)):
        fmt = "SPEAKER x 1 {:.2f} {:.2f} <NA> <NA> {:s} <NA> <NA>"
        rttm += f"{fmt.format(float(segments[i].stime), float(segments[i].etime) - float(segments[i].stime), str(segments[i].spkr))}\n"

    return rttm