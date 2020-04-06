import pickle
import re
import time
import pendulum
import hashlib
import numpy as np
import pandas as pd
import json
import os
import glob
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from dateutil import parser

from multiprocessing import Pool

_RESEARCH_PAPERS_SAVE_FILE = 'ResearchPapers.pickle'

_abstract_terms_ = '(Publisher|Abstract|Summary|BACKGROUND|INTRODUCTION)'

# Some titles are is short and unrelated to viruses
# This regex keeps some short titles if they seem relevant
_relevant_re_ = '.*vir.*|.*sars.*|.*mers.*|.*corona.*|.*ncov.*|.*immun.*|.*nosocomial.*'
_relevant_re_ = _relevant_re_ + '.*epidem.*|.*emerg.*|.*vacc.*|.*cytokine.*'


def remove_common_terms(abstract):
    return re.sub(_abstract_terms_, '', abstract)


def start(data):
    return data.copy()


def clean_title(data):
    # Set junk titles to NAN
    title_relevant = data.title.fillna('').str.match(_relevant_re_, case=False)
    title_short = data.title.fillna('').apply(len) < 30
    title_junk = title_short & ~title_relevant
    data.loc[title_junk, 'title'] = ''
    return data


def show_common(data, column, head=20):
    common_column = data[column].value_counts().to_frame()
    common_column = common_column[common_column[column] > 1]
    return common_column.head(head)


def clean_abstract(data):
    # Set unknowns to NAN
    abstract_unknown = data.abstract == 'Unknown'
    data.loc[abstract_unknown, 'abstract'] = np.nan

    data['no_abstract'] = data.abstract.isna()

    # Fill missing abstract with the title
    data.abstract = data.abstract.fillna(data.title)

    # Remove common terms like publisher
    data.abstract = data.abstract.fillna('').apply(remove_common_terms)

    # Remove the abstract if it is too common
    common_abstracts = show_common(data, 'abstract').query('abstract > 2') \
        .reset_index().query('~(index =="")')['index'].tolist()
    data.loc[data.abstract.isin(common_abstracts), 'abstract'] = ''

    return data


def date_diff(date):
    if pd.isnull(date):
        return ''
    timestamp = date.timestamp()
    if timestamp < 0:
        diff = f'more than {pendulum.from_timestamp(0).diff_for_humans()}'
    else:
        diff = pendulum.from_timestamp(timestamp).diff_for_humans()
    return diff


def add_date_diff(data, date_column='publish_time', new_column='when'):
    data[new_column] = data[date_column].apply(date_diff)
    return data


def drop_missing(data):
    missing = (data.publish_time.isnull()) & \
              (data.sha.isnull()) & \
              (data.title == '') & \
              (data.abstract == '')
    return data[~missing].reset_index(drop=True)


def fill_nulls(data):
    data.authors = data.authors.fillna('')
    data.doi = data.doi.fillna('')
    data.journal = data.journal.fillna('')
    data.abstract = data.abstract.fillna('')
    return data


def clean_metadata(metadata):
    print('Cleaning metadata')
    return metadata.pipe(start) \
        .pipe(clean_title) \
        .pipe(clean_abstract) \
        .pipe(add_date_diff) \
        .pipe(drop_missing) \
        .pipe(fill_nulls)


class Cord19():
    """
    Transforms raw csv and json files into pandas database.
    metadata : ['cord_uid', 'sha', 'source_x', 'title', 'doi', 'pmcid', 'pubmed_id',
       'license', 'abstract', 'publish_time', 'authors', 'journal',
       'Microsoft Academic Paper ID', 'WHO #Covidence', 'has_pdf_parse',
       'has_pmc_xml_parse', 'full_text_file', 'url', 'when', 'no_abstract']

    df_docs : ['cord_uid', 'lsid', 'gsid', 'Name', 'Text', 'Subtype']

    'cord_uid' is unique for metadata. Can be used for section lookup in df_docs

    """
    def __init__(self, directory, output_dir=None):
        self.directory = directory

        # Default path if not provided
        if not output_dir:
            output_dir = os.path.join(os.path.expanduser("~"), "Documents",
                                      "cord19m", "models")
        self.output_dir = output_dir
        print('output_dir', output_dir)

        # Create if output path doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        self.load_metadata()

    def load_metadata(self):
        """
        Read and clean metadata file to dataframe
        """
        metadata_path = os.path.join(self.directory, "metadata.csv")
        print('Loading metadata from', metadata_path)

        dtypes = {'Microsoft Academic Paper ID': 'str', 'pubmed_id': str}
        metadata = pd.read_csv(metadata_path,
                               dtype=dtypes,
                               parse_dates=['publish_time'])
        metadata = metadata.drop_duplicates(subset=['cord_uid'])
        self.total_docs = len(metadata.index)
        self.metadata = clean_metadata(metadata)

    def load_json_document(self, row):
        cord_uid = row['cord_uid']

        # list of [section_name, section_text, source of text]
        sections = []
        # for avoiding duplicates, where multiple sha ids available for single cord_uid
        # or both pmc_json and pdf_json available
        sections_keys = {}

        # Add title and abstract sections
        for name in ["title", "abstract"]:
            text = row[name]
            if text:
                # Remove leading and trailing []
                text = re.sub(r"^\[", "", text)
                text = re.sub(r"\]$", "", text)
                text = re.sub(_abstract_terms_, '', text)
                sections.append([name.upper(), text, 'metadata'])

        # Get information of json files
        subset = row["full_text_file"]
        fnames = [["pmc_json", row["pmcid"] +
                   ".xml"]] if row["pmcid"] == row["pmcid"] else []
        shas = row["sha"].split("; ") if row["sha"] == row["sha"] else None
        if shas:
            fnames += [["pdf_json", sha] for sha in shas]

        if fnames and subset == subset:
            for [subtype, fname] in fnames:
                # Build article path. Path has subset directory twice.
                article = os.path.join(self.directory, subset, subset, subtype,
                                       fname + ".json")

                try:
                    if os.path.isfile(article):
                        with open(article) as jfile:
                            data = json.load(jfile)

                            # Extract text from each section
                            for section in data["body_text"]:

                                # Section name
                                name = section["section"].upper() if len(
                                    section["section"].strip(
                                    )) > 0 else 'NO_NAME'

                                if name not in sections_keys:
                                    sections_keys[name] = [
                                        len(sections), fname
                                    ]
                                    sections.append(
                                        [name, section["text"], subtype])

                                # if text from same subtype, concatenate it
                                elif fname == sections_keys[name][1]:
                                    sections[sections_keys[name]
                                             [0]][1] += '\n' + section["text"]

                except Exception as ex:
                    print("Error processing text file: {}".format(article), ex)

        df_doc = pd.DataFrame(sections, columns=['Name', 'Text', 'Subtype'])
        df_doc['cord_uid'] = cord_uid
        df_doc['lsid'] = range(1, len(df_doc) + 1)

        return df_doc

    def load_text_data(self, cord_uids=None):
        """
        Read json files to create df_docs dataframe.

        Args:
            cord_uids (optional): will read only selected files 
        """

        out_path = os.path.join(self.output_dir, 'df_docs.xlsx')

        if not cord_uids and os.path.isfile(out_path):
            # check if processed file available
            start = time.time()
            print('loading from saved file', out_path)
            self.df_docs = pd.read_excel(out_path)
            print('loaded in %0.2fs' % (time.time() - start))
            return
        elif not cord_uids and not os.path.isfile(out_path):
            # processed complete metadata
            metadata_filtered = self.metadata
        else:
            # processed given cord_uids
            metadata_filtered = self.metadata[self.metadata.cord_uid.isin(
                cord_uids)]

        df_docs = []
        for index, row in tqdm(self.metadata.iterrows(),
                               total=self.total_docs):

            df_docs.append(self.load_json_document(row))

        df_docs = pd.concat(df_docs)

        df_docs['gsid'] = range(1, len(df_docs) + 1)
        self.df_docs = df_docs[[
            'cord_uid', 'lsid', 'gsid', 'Name', 'Text', 'Subtype'
        ]]

        if not cord_uids:
            print('Saving file')
            self.df_docs.to_excel(out_path, index=False)

    def get_paper_by_id(self, cord_uid):
        """
        Return metadata(series) and text data(dataframe) by cord_uid

        Args:
            cord_uids: id of document
        """
        try:
            return self.metadata[self.metadata['cord_uid'] ==
                                 cord_uid].T.iloc[:, 0], self.df_docs[
                                     self.df_docs['cord_uid'] == cord_uid]
        except Exception as e:
            print(
                "Error fetching data for cord_uid: {}.\nCheck if text data loaded properly"
                .format(cord_uid), e)
            return None, None


if __name__ == '__main__':
    x = Cord19('../../../data/')
    x.load_text_data()