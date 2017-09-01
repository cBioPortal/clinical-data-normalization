### load in packages
import argparse
import seaborn
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Levenshtein
import scipy
import urllib, json
from collections import Counter
import math
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

#define some functions
#-----------------------------------
#downloads main study information from the portal
def get_all_study_info():
    url = "http://www.cbioportal.org/api/studies"
    response = urllib.urlopen(url)
    json_data = json.loads(response.read())
    return json_data

#downloads all attribute data for a list of studies
def get_attribute_data(study_list):
    studies=[]
    study_attributes = []

    #loop over studies and download data from api
    for study in study_list:
        try:
            studyID = study['studyId']
        except:
            studyID = study
        study_api_url = 'http://cbioportal.org/api/studies/' + studyID + '/clinical-attributes'
        df = pd.read_json(study_api_url)

        #make sure the study is not empty
        if not df.empty:
            studies.append((studyID ,df))
            study_attributes.append((studyID ,df['clinicalAttributeId'].tolist()))
            
    return studies, study_attributes

def get_clinical_data_values(study_list, attribute_type='PATIENT'):
    studies_clinical_data=[]
    study_attributes = []

    #loop over studies and download data from api
    for study in study_list:
        try:
            studyID = study['studyId']
        except:
            studyID = study
        study_api_url = 'http://cbioportal.org/api/studies/' + studyID + '/clinical-data?clinicalDataType=' + attribute_type
        df = pd.read_json(study_api_url)
        #make sure the study is not empty
        if not df.empty:
            df=df.pivot(values='value', index='entityId', columns='clinicalAttributeId').fillna(value=np.nan)
            studies_clinical_data.append((studyID ,df))
            #study_attributes.append((studyID ,df['clinicalAttributeId'].tolist()))
    return studies_clinical_data

def get_study_names(study_attribute_list):
    names=[]
    for study in study_attribute_list:
        names.append(study[0])
    return names

def study_attributes_to_df(study_attribute_data):
    #transform data into boolean table of studies/attributes
    #entries in the table are 1 if a study contains an attribute and 0 otherwise
    study_attribute = []
    for i in study_attribute_data:
        for j in i[1]:
            study_attribute.append((i[0],j))

    study_attribute_pairs = pd.DataFrame.from_records(study_attribute, columns = ['study','attribute'])

    study_data_nolabel = pd.get_dummies(study_attribute_pairs['attribute'])

    study_data_combined = pd.concat([study_attribute_pairs['study'], study_data_nolabel], axis=1)
    study_data_combined = study_data_combined.groupby('study').sum()
        
    return study_data_combined

def drop_study(study_to_drop, combined_data):
    combined_data.drop(study_to_drop, axis=0, inplace=True)
    combined_data.drop([col for col, val in combined_data.sum().iteritems() if val == 0], axis=1, inplace=True)
    return combined_data

def find_attribute_name_matches(cBioPortal_attributes, new_attributes, cutoff=0.9):
    all_col_names = list(cBioPortal_attributes)
    lev_dist = np.zeros([len(all_col_names), len(new_attributes)])
    for i in range(len(all_col_names)):
        for j in range(len(new_attributes)):
            lev_dist[i,j]=Levenshtein.ratio(unicode(all_col_names[i].upper()), unicode(new_attributes[j].upper()))
            
    all_lev_distances = pd.DataFrame(data=lev_dist.T, index=new_attributes, columns=all_col_names)
    matches = pd.DataFrame(data=list(all_lev_distances[all_lev_distances > cutoff].stack().index))
            
    return matches
        
def plot_attribute_distribution(cBioPortal_data, new_study_data):
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.rcParams.update({'font.size':20})
    plt.figure()
    ax=seaborn.distplot(cBioPortal_data.sum(axis=1), kde=False)
    ax.set(ylabel='number of studies', xlabel='attributes in study')
    plt.axvline(len(new_study_data), color='k', linestyle='dashed', linewidth=2)
    plt.savefig('n_attribute_distribution.png')
    plt.close()

def plot_unique_and_common_attribute_distributions(cBioPortal_data, non_matching_attributes):
    #unique attributes plot
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.rcParams.update({'font.size':20})
    plt.figure()
    unique_attributes_cBio_studies = cBioPortal_data.T[(cBioPortal_data.sum(axis=0)==1).values].sum(axis=0)
    ax=seaborn.distplot(unique_attributes_cBio_studies, kde=False)
    ax.set(ylabel='number of studies', xlabel='unique attributes in study')
    plt.axvline(non_matching_attributes.size, color='k', linestyle='dashed', linewidth=2)
    plt.savefig('n_unique_attribute_distribution.png')
    plt.close()
    
    #common attributes plot
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.rcParams.update({'font.size':20})
    plt.figure()
    common_attributes = cBioPortal_data.sum(axis=1) - unique_attributes_cBio_studies
    ax=seaborn.distplot(common_attributes, kde=False)
    ax.set(ylabel='number of studies', xlabel='common attributes in study')
    plt.axvline(exact_matching_attributes.size, color='k', linestyle='dashed', linewidth=2)
    plt.savefig('n_common_attribute_distribution.png')
    plt.close()
    
def get_new_study_attributes(test_study_data):
    if "PATIENT_ID" in test_study_data:
        del test_study_data["PATIENT_ID"]
    if "SAMPLE_ID" in test_study_data:   
        del test_study_data["SAMPLE_ID"]
    test_study_attribute_names = map(unicode,map(str.upper,list(test_study_data)))
    return test_study_attribute_names

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    
    try:
        import unicodedata
        unicodedata.numeric(s)
    except (TypeError, ValueError):
        pass
    
    return False

def process_clinical_data(all_study_clinical_data, study_to_drop=''):
    attribute_data = []
    for study in all_study_clinical_data:
        if study[0] != study_to_drop:
            study_data_attributes = list(study[1])

            #get attribute data to filter based on datatype
            if api_flag:
                study_api_url = 'http://cbioportal.org/api/studies/' + study[0] + '/clinical-attributes'
                df = pd.read_json(study_api_url)
                for attribute in study_data_attributes:
                    if not attribute.endswith('_ID'):
                        if df['datatype'][df['clinicalAttributeId']==attribute].values[0] == 'STRING':
                            data = set(study[1][attribute])
                            for d in data:
                                if not is_number(d):
                                    attribute_data.append((attribute.upper(), d.upper()))
            
            else:
                for attribute in study_data_attributes:
                    if (not unicode(attribute).endswith("ID")) and ("FILE_NAME" not in attribute)  and ("DATE" not in attribute):
                        data = set(study[1][attribute])
                        for d in data:
                            if not is_number(d):
                                attribute_data.append((attribute.upper(), d.upper()))

    attribute_data_pairs = pd.DataFrame.from_records(attribute_data, columns = ['attribute','data'])
    attribute_data_nolabel = pd.get_dummies(attribute_data_pairs['data'])
    attribute_data_combined = pd.concat([attribute_data_pairs['attribute'], attribute_data_nolabel], axis=1)
    attribute_data_combined = attribute_data_combined.groupby('attribute').sum()
    attribute_data_combined[attribute_data_combined>0]=1

    #drop values which only occur in single attribute
    attribute_data_combined.drop([col for col, val in attribute_data_combined.sum().iteritems() if val > 10], axis=1, inplace=True)
    attribute_data_combined.drop([row for row, val in attribute_data_combined.sum(axis=1).iteritems() if val < 1], axis=0, inplace=True)
    
    return attribute_data_combined

def process_new_study_data(new_study_data):
    ns_attribute_data = []
    for attribute in list(new_study_data):
        if (attribute != u'SAMPLE_ID') and (attribute != u'PATIENT_ID'):
            data = set(new_study_data[attribute])
            for d in data:
                if not is_number(d):
                    ns_attribute_data.append((attribute.upper(), d.upper()))
    
    ns_attribute_data_pairs = pd.DataFrame.from_records(ns_attribute_data, columns = ['attribute','data'])

    ns_attribute_data_nolabel = pd.get_dummies(ns_attribute_data_pairs['data'])

    ns_attribute_data_combined = pd.concat([ns_attribute_data_pairs['attribute'], ns_attribute_data_nolabel], axis=1)
    ns_attribute_data_combined = ns_attribute_data_combined.groupby('attribute').sum()

    ns_attribute_data_combined[ns_attribute_data_combined>0]=1
    ns_attribute_data_combined = ns_attribute_data_combined.T.add_prefix('NEW_STUDY_').T
    
    return ns_attribute_data_combined

def get_clusters(attribute_value_data):
    data_link = scipy.cluster.hierarchy.linkage(attribute_value_data, method='complete', metric='cosine') # computing the linkage
    thresh = 0.7*max(data_link[:,2])
    np.clip(data_link,0,np.max(data_link),data_link)
    clusters = fcluster(data_link, thresh, 'distance')
    clustered_attributes=[]

    for label in np.unique(clusters):
        clustered_attributes.append(list(np.asarray(attribute_value_data.index.values)[clusters==label]))
        
    plt.rcParams['figure.figsize'] = (180.0, 25.0)
    plt.rcParams.update({'font.size':40})
    plt.figure()
    den=dendrogram(data_link, labels = combined_attribute_values.index, leaf_font_size = 15, above_threshold_color='#AAAAAA')
    plt.savefig('dendrogram.png', bbox_inches='tight')
    plt.close()
        
    return clustered_attributes

def output_cluster_matches(all_clusters):
    print ""
    print "================================================"
    print "possible attribute matches based on data values"
    print "================================================"


    for cluster in all_clusters:
        if len(cluster)>1:
            for attribute in cluster:
                if 'NEW_STUDY_' in attribute:
                    matches = np.setdiff1d(cluster, attribute)
                    match_count = 0
                    for match in matches:
                        if 'NEW_STUDY_' not in match:
                            match_count += 1
                            if match_count == 1:
                                print "================================================"
                                print "possible matches for: " + attribute
                                print "--------------------"
                            print "possible match: " + match

def find_data_files(basedir):
    import os
    import os.path

    file_list = []
    for dirpath, dirnames, filenames in os.walk(basedir):
        for filename in [f for f in filenames if ((("clinical" in f) and ("data" in f)) and (f.endswith(".txt")))]:
            file_list.append(os.path.join(dirpath, filename))
    return file_list

def load_data_from_files(file_list, basedir):
    study_data = []
    attribute_data = []
    for clinical_file in file_list:
        #load in individual data from files and append to a list
        df = pd.read_table(clinical_file, skiprows=0)
        if (list(df)[0]!='SAMPLE_ID' and list(df)[0]!='PATIENT_ID'):
            rows_to_skip=df[(df[df.columns[0]]=="SAMPLE_ID") | (df[df.columns[0]]=="PATIENT_ID") | (df[df.columns[0]]=="OTHER_PATIENT_ID") | (df[df.columns[0]]=="OTHER_SAMPLE_ID")].index[0]+1
            df = pd.read_table(clinical_file, skiprows=rows_to_skip)
        df.columns = map(str.upper, df.columns)
        col_names = list(df)
        tcga_provisional_flag = False
        study_name = clinical_file.split(basedir+'/')[1].split('/')[0]
        if clinical_file.split(basedir+'/')[1].split('/')[1] == 'tcga':
            study_name = study_name + '_tcga'
        if study_name == '':
            study_name = clinical_file.split(basedir+'/')[1].split('/')[1]
            if clinical_file.split(basedir+'/')[1].split('/')[2] == 'tcga':
                study_name = study_name + '_tcga'
        study_data.append((study_name, col_names))
        attribute_data.append((study_name, df))
    return study_data, attribute_data

def process_levenshtein_matches(exact_matches, possible_matches, non_matching_attributes):
    match_pairs = []
    no_matches = []
    for attribute in non_matching_attributes:
        try:
            filtered_matches = np.setdiff1d(possible_matches[possible_matches[0]==attribute][1].values, exact_matches)
        except:
            filtered_matches = np.empty(0)
        if (filtered_matches.size)>0:
            for matching_attribute in filtered_matches:
                match_pairs.append((attribute, matching_attribute))
        else:
            no_matches.append(attribute)
    return match_pairs, no_matches

def process_cluster_matches(all_clusters):
    match_pairs = []
    for cluster in all_clusters:
        if len(cluster)>1:
            for attribute in cluster:
                if 'NEW_STUDY_' in attribute:
                    matches = np.setdiff1d(cluster, attribute)
                    for match in matches:
                        if 'NEW_STUDY_' not in match:
                            match_pairs.append((attribute[10:], match))
    return match_pairs

def get_unique_attributes(unique_attributes, value_matches):
    no_matches = []
    for attribute in unique_attributes:
        if sum(value_matches['New study attribute']==attribute)==0:
            no_matches.append(attribute)
    return no_matches

def generate_latex_report(match_table, predicted_types, study_name):
    from pylatex import NoEscape, Document, Section, Subsection, Tabular, Math, TikZ, Axis, Plot, Figure, Matrix, Alignat
    from pylatex.utils import italic
    import os

    file_directory = ''

    geometry_options = {"tmargin": "1in", "lmargin": "1in", "bmargin": "1in", "rmargin": "1in"}
    doc = Document(geometry_options=geometry_options)

    with doc.create(Section('cBioPortal new study report')):
        doc.append('Report for study: ' + study_name)

        with doc.create(Subsection('Attribute matches')):
            doc.append('  Below are the possible matches between attributes from existing data on the cBioPortal and the new study.')
            doc.append('  The metric used to detect each match is denoted by the symbols follwing the attribute name of the match.')
            doc.append('  Additionally, the number of studies in which the matching attribute occurs is given to indicate how popular the attribute is among existing studies.')
            if predicted_types is not None:
                doc.append('  In the second table, predictions are given as to whether an attribute is a patient or sample attribute.')
                doc.append('  The sample/patient prediction is based on what is most common for that particular attribute in the existing cBioPortal studies.')
            doc.append(NoEscape(r'\\'))
            doc.append(NoEscape(r'\\'))

            with doc.create(Tabular('|c|c|')) as table:
                table.add_hline()
                table.add_row(list(match_table))
                table.add_hline()
                table.add_hline()

                for row in match_table.index:
                    table.add_row(list(match_table.loc[row,:]))
                    table.add_hline()
        doc.append(NoEscape(r'\string^ represents matches found based on the attribute names\\'))
        doc.append(NoEscape(r'\string* represents matches found based on clustering of the attribute values\\'))
        doc.append(NoEscape(r'NOTE: PATIENT_ID and SAMPLE_ID are omitted as they should be present in every study.\\'))

        if predicted_types is not None:
            with doc.create(Subsection('Matching attribute types')):
                with doc.create(Tabular('|c|c|')) as table:
                    table.add_hline()
                    table.add_row(list(predicted_types))
                    table.add_hline()
                    table.add_hline()

                for row in predicted_types.index:
                    table.add_row(list(predicted_types.loc[row,:]))
                    table.add_hline()

        doc.append(NoEscape(r'\newpage'))

        with doc.create(Subsection('Number of attribute distributions')):
            with doc.create(Figure(position='h!')) as n_attributes:
                n_attributes.add_image(file_directory + 'n_attribute_distribution.png', width=NoEscape(r'0.99\textwidth'))
                n_attributes.add_caption('Comparison between the number of attributes in the new study and the number of attributes in each existing study on cBioPortal.  The dashed black line indicates the number of attributes in the new study, while the histogram shows the data for existing cBioPortal studies.')

        #with doc.create(Subsection('Number of unique attributes')):
            with doc.create(Figure(position='h!')) as n_unique_attributes:
                n_unique_attributes.add_image(file_directory + 'n_unique_attribute_distribution.png', width=NoEscape(r'0.99\textwidth'))
                n_unique_attributes.add_caption('Comparison between the number of unique attributes in the new study and the number of unique attributes in each existing study on cBioPortal.  The dashed black line indicates the number of unqiue attributes in the new study, while the histogram shows the data for existing cBioPortal studies.')

        #with doc.create(Subsection('Number of common attributes')):
            with doc.create(Figure(position='h!')) as n_common_attributes:
                n_common_attributes.add_image(file_directory + 'n_common_attribute_distribution.png', width=NoEscape(r'0.99\textwidth'))
                n_common_attributes.add_caption('Comparison between the number of common attributes in the new study and the number of common attributes in each existing study on cBioPortal.  The dashed black line indicates the number of common attributes in the new study, while the histogram shows the data for existing cBioPortal studies.')

    doc.generate_pdf('report', clean_tex=True)

def combine_matches(exact, name, value, no_matches):   
    all_matches = (pd.concat([exact, value, name, no_matches])
                      .drop_duplicates()
                      .sort_values('New study attribute',axis=0)
                      .reset_index(drop=True))
    
    for index, row in all_matches.iterrows():
        append_string = ''
        if (sum(exact['Possible matches'] == row['Possible matches']) > 0) or (sum(name['Possible matches'] == row['Possible matches']) > 0):
            append_string += '^'
        if sum(value['Possible matches'] == row['Possible matches']) > 0:
            append_string += '*'
        else:
            append_string += ''

        if row['Possible matches'] != 'No matches found':            
            all_matches['Possible matches'][index] = all_matches['Possible matches'][index] + append_string + " (found in " + str(int(study_data_combined.sum()[row['Possible matches']])) + " other studies)"

    return all_matches

def attribute_type_prediction(exact_matches, name_matches, value_matches, studies):
    sample_or_patient = {True:"Patient", False:"Sample"}
    match_type_table = []
    only_matches = (pd.concat([exact_matches, value_matches, name_matches])
                      .drop_duplicates()
                      .sort_values('New study attribute',axis=0)
                      .reset_index(drop=True))
    
    for matching_attribute in only_matches['Possible matches'].values:
        studies_with_attribute=study_data_combined[study_data_combined[matching_attribute]>0].index    
        attribute_ps_types=[]
        for study in studies_with_attribute:
            study_index=study_names.index(study)

            attribute_data=studies[study_index][1]
            attribute_ps_type=attribute_data[attribute_data['clinicalAttributeId']==matching_attribute]['patientAttribute']
            attribute_ps_types.append(sample_or_patient[attribute_ps_type.get_values()[0]])

        match_type_table.append((matching_attribute, max(Counter(attribute_ps_types))))
    return pd.DataFrame(match_type_table, columns=["Matching Atrribute","Patient or Sample prediction"])

def generate_html_report(match_table, predicted_types, study_name):
    #open file to write
    f = open('report.html', 'w')
    #f.write('<center>')
    f.write('<center><font size=\"6\">cBioPortal new study report</font></center> <br>\n')
    f.write('Report for study: ' + study_name + "</font></center><br>\n")

    f.write('<center><font size=\"5\">Attribute matches</font></center><br>\n')

    f.write('Below are the possible matches between attributes from existing data on the cBioPortal and the new study.\n')
    f.write('  The metric used to detect each match is denoted by the symbols follwing the attribute name of the match.\n')
    f.write('  Additionally, the number of studies in which the matching attribute occurs is given to indicate how popular the attribute is among existing studies.\n')
    f.write('<br>\n')
    f.write('<br>\n')

    f.write(match_table.to_html(index=False))

    f.write('^ represents matches found based on the attribute names<br>')
    f.write('* represents matches found based on clustering of the attribute values<br>')
    f.write('NOTE: PATIENT_ID and SAMPLE_ID are omitted as they should be present in every study.')
    f.write('<br>\n')
    f.write('<br>\n')

    if predicted_types is not None:
        f.write('<center><font size=\"5\">Matching attribute types </font></center><br>\n')
        f.write('Predictions are given as to whether an attribute is a patient or sample attribute.\n')
        f.write('  The sample/patient prediction is based on what is most common for that particular attribute in the existing cBioPortal studies.\n')
        f.write('<br>\n')
        f.write(predicted_types.to_html(index=False)) 
        f.write('<br>\n')    

    #with doc.create(Subsection('Number of attributes distribution\n')):
    f.write('<div class="image" style="display:table;">')
    f.write('<img src=\"' + file_directory + 'n_attribute_distribution.png\"' + 'alt=\"number of attributes\" width=\"800\">')
    f.write('<br>\n')
    f.write('<div style="display:table-caption;caption-side:bottom;">Comparison between the number of attributes in the new study and the number of attributes in each existing study on cBioPortal.  The dashed black line indicates the number of attributes in the new study, while the histogram shows the data for existing cBioPortal studies.</div>\n')
    f.write('</div>\n')

    f.write('<br>')

    f.write('<div class="image" style="display:table;">')
    f.write('<img src=\"' + file_directory + 'n_unique_attribute_distribution.png\"' + 'alt=\"number of unique attributes\" width=\"800\">')
    f.write('<br>\n')
    f.write('<div style="display:table-caption;caption-side:bottom;">Comparison between the number of unique attributes in the new study and the number of unique attributes in each existing study on cBioPortal.  The dashed black line indicates the number of unqiue attributes in the new study, while the histogram shows the data for existing cBioPortal studies.</div>\n')
    f.write('</div>\n')

    f.write('<br>')

    f.write('<div class="image" style="display:table;">')
    f.write('<img src=\"' + file_directory + 'n_common_attribute_distribution.png\"' + 'alt=\"number of common attributes\" width=\"800\">')
    f.write('<br>\n')
    f.write('<div style="display:table-caption;caption-side:bottom;">Comparison between the number of common attributes in the new study and the number of common attributes in each existing study on cBioPortal.  The dashed black line indicates the number of common attributes in the new study, while the histogram shows the data for existing cBioPortal studies.</div>\n')
    f.write('</div>\n')

    f.close()

#values that should be read in
# Parse arguments
file_directory=''

parser = argparse.ArgumentParser()
parser.add_argument("--new_study_path", help="path to new study data")
parser.add_argument("--study_to_drop", help="if study being tested is already on cBioPortal it may be helpful to drop that study from the analysis")
parser.add_argument("--specific_study", help="name of specific study on the portal to test")
parser.add_argument("--datahub_path", help="directory where the datahub is located")
parser.add_argument("--output_pdf", action='store_true', help="flag to output pdf")
args = parser.parse_args()

new_study_path = args.new_study_path
study_to_drop = args.study_to_drop
specific_study = args.specific_study
datahub_path = args.datahub_path
output_pdf = args.output_pdf
random_study=False
if (new_study_path is None) and (specific_study is None):
    random_study = True

api_flag=True
if datahub_path is not None:
    api_flag = False
    print "Using cBioPortal data from local datahub: " + datahub_path
else:
    print "cBioPortal data will be obtained via API"

latex_flag=False
if output_pdf:
    latex_flag=True
    print "PDF file will be output"


#main part of the script below
#main function
if api_flag:
    all_cBioPortalStudies = get_all_study_info()
    studies, study_attributes = get_attribute_data(all_cBioPortalStudies)
else:
    list_of_files = find_data_files(datahub_path)
    study_attributes, attribute_value_data = load_data_from_files(list_of_files, datahub_path)
study_names = get_study_names(study_attributes)
study_data_combined = study_attributes_to_df(study_attributes)


#choose a study to serve as a "new" study
if random_study or specific_study is not None:
    if random_study:
        test_study = random.choice(study_names)
    else:
        test_study = specific_study
    print "Study being analyzed: " + test_study
    study_to_drop = test_study
    study_data_combined = drop_study(test_study, study_data_combined)
    test_study_data, test_study_clin_attributes = get_attribute_data([test_study])
    test_study_attribute_data = test_study_data[0][1]
    test_study_attribute_names = test_study_attribute_data['clinicalAttributeId'].values
    test_study_patient_attribute_values = get_clinical_data_values([test_study], attribute_type="PATIENT")
    test_study_sample_attribute_values = get_clinical_data_values([test_study], attribute_type="SAMPLE")
    processed_test_study_patient_attribute_values = process_new_study_data(test_study_patient_attribute_values[0][1])
    processed_test_study_sample_attribute_values = process_new_study_data(test_study_sample_attribute_values[0][1])
    new_study_clinical_attribute_values = pd.concat([processed_test_study_patient_attribute_values,
                                                     processed_test_study_sample_attribute_values],
                                                     axis=0).fillna(value=0)

#otherwise read in data from a new study
else:
    print "Study being analyzed: " + new_study_path
    new_study_data = pd.read_table(new_study_path)
    test_study_attribute_names = get_new_study_attributes(new_study_data)
    if study_to_drop is not None:
        study_data_combined = drop_study(study_to_drop, study_data_combined)
    new_study_clinical_attribute_values = process_new_study_data(new_study_data)
    
#check matching attributes (via name only)
exact_matching_attributes = np.intersect1d(test_study_attribute_names, list(study_data_combined))
non_matching_attributes = np.setdiff1d(test_study_attribute_names, list(study_data_combined))
possible_matches = find_attribute_name_matches(study_data_combined, non_matching_attributes)

#check matching attributes based on data values
if api_flag:
    patient_data = get_clinical_data_values(all_cBioPortalStudies, attribute_type="PATIENT")
    patient_data_processed = process_clinical_data(patient_data, study_to_drop)
    sample_data = get_clinical_data_values(all_cBioPortalStudies, attribute_type="SAMPLE")
    sample_data_processed = process_clinical_data(sample_data, study_to_drop)
    combined_PS_attribute_data = pd.concat([patient_data_processed, sample_data_processed], axis=0).fillna(value=0)
else:
    combined_PS_attribute_data = process_clinical_data(attribute_value_data, study_to_drop)
combined_attribute_values = pd.concat([combined_PS_attribute_data, new_study_clinical_attribute_values], axis=0).fillna(value=0)

#get clusters and plot dendrogram
clusters = get_clusters(combined_attribute_values)

#process matches
processed_name_matches, no_name_matches = process_levenshtein_matches(exact_matching_attributes, 
                                                                      possible_matches,
                                                                      non_matching_attributes)
processed_value_matches = process_cluster_matches(clusters)

#combine matches into one dataframe
exact_matches_df = pd.DataFrame(np.column_stack([exact_matching_attributes, exact_matching_attributes]), 
                                columns=['New study attribute', 'Possible matches'])
name_matches_df = pd.DataFrame(processed_name_matches, columns=['New study attribute', 'Possible matches'])
value_matches_df = pd.DataFrame(processed_value_matches, columns=['New study attribute', 'Possible matches'])
no_name_or_value_matches = get_unique_attributes(no_name_matches, value_matches_df)
no_matches_df = pd.DataFrame(np.asarray(([no_name_or_value_matches, ['No matches found']*len(no_name_or_value_matches)])).T,
                             columns=['New study attribute', 'Possible matches'])
all_matches_df = combine_matches(exact_matches_df, name_matches_df, value_matches_df, no_matches_df)

predicted_attribute_types=None
if api_flag:
    predicted_attribute_types = attribute_type_prediction(exact_matches_df, name_matches_df, value_matches_df, studies)

if new_study_path != None:
    test_study = new_study_path
#print all_matches_df
if latex_flag:
    generate_latex_report(all_matches_df, predicted_attribute_types, test_study)
    print "Results written to report.pdf"
else:
    generate_html_report(all_matches_df, predicted_attribute_types, test_study)
    print "Results written to report.html"
    #print "report for study: " + test_study
    #print all_matches_df
    #print '^ represents matches found based on the attribute names'
    #print '* represents matches found based on clustering of the attribute values'
    #if predicted_attribute_types is not None:
    #    print predicted_attribute_types

#make and save figures
plot_attribute_distribution(study_data_combined, test_study_attribute_names)
plot_unique_and_common_attribute_distributions(study_data_combined, non_matching_attributes)
