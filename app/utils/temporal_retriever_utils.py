import datetime


def parse_datetime(dtime):
    d_time = dtime.split("-")
    return datetime.datetime(day=int(d_time[2]),
                             month=int(d_time[1]),
                             year=int(d_time[0]))
