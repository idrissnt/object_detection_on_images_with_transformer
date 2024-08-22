from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from ArchiMedConnector.A3_Connector import A3_Connector # import Connector

def get_data_from_archimed(study_code = '2020-128', dir_to_save = "/home/pyuser/data_1/data_archimed_/"):
    a3conn= A3_Connector()

    exams = a3conn.getExams(filterStr=f"study.studyCode = '{study_code}'")
    n = 0
    for exam in exams:
        # print(exam["examCode"])

        examinfos = a3conn.getExamFullInfos(
            exam["examCode"],  # Exam Code
            # worklistType = 'Serie'
        )

        fname =  exam["examCode"]
        a3conn.downloadFiles(
            examinfos,  # files ids (see node infos)
            destDir=f"{dir_to_save}{fname}/",
            progress=False
        )


