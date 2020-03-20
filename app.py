# -*- coding: utf-8 -*-
# import supporting libraries
import pandas as pd
import numpy as np
import json
import requests
import io
import time
import re

# import dash visualization libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# av calculation logic
def apply_copays(service_count,payment_pattern,avg_service_cost=0):
    # this logic limits the copay applied to the avg cost per service
    if avg_service_cost>0:
        payment_pattern = np.minimum(payment_pattern,avg_service_cost)
    return payment_pattern[0:service_count].sum()

def calculateAV(design,population,pop_size):
    # set random seed
    np.random.seed(49)
    
    #time this part of the code
    start_time = time.time()
        
    # set up empty list for calculated AVs
    AV_list = []
    
    # define elements of plan design to model
    pd_elements = {'ER_visit','OP_observation','OP_surgery','Allergy', \
                   'Ambulance_transportation','Anesthesia','Chiropractic', \
                   'Consultations','DME_prosthetics_supplies','Hospice', \
                   'IP_visit','Immunizations','Misc_service','OP_misc', \
                   'OP_pathology','OP_radiology','Ophthamology','PCP_visits', \
                   'Physical Medicine','Prev_visits','Prof_ER','Prof_adm_drugs', \
                   'Prof_cardiovascular','Prof_inpatient','Prof_other', \
                   'Prof_pathology','Prof_radiology','Prof_surgery','Psychiatry', \
                   'Rx_brand','Rx_generic','SNF','Specialist_visits'}
    
    grp_clmnts = clmnt_df.groupby(['age_group','gender'])
    grp_pop = pd.DataFrame(population).groupby(['agegroup','gender'])
    
    clmnt_df['allowed']=0
    clmnt_df['deductible']=0
    clmnt_df['copays']=0
    clmnt_df['coin']=0
    
    deductible = int(design['ind_ded'].replace('$',''))
    coinsurance = float(design['coin'].replace('%',''))
    if coinsurance > 1: coinsurance = coinsurance/100
    moop = int(design['ind_oop'].replace('$',''))
               
    ## loop through elements and apply cost sharing provision
    for element in pd_elements:
        counting_element = 'services'
        if element in ['IP','ER_visit','OP_observation','OP_surgery']:
            counting_element = 'visits'
        services = clmnt_df[element+'('+ counting_element + ')'].fillna(0)
        allowed = clmnt_df[element+'(allowed)'].fillna(0)
        avg_allowed_per_service = np.divide(allowed,services, \
                                            out=np.zeros_like(allowed), \
                                                where=services!=0)
        cost_share = design[element]
        #remove allowed and services if not covered category
        if (cost_share.find("X")!=-1):
            #quickest solution is to just skip rest of code and continue loop
            continue
        #apply any limit to services 
        limit = 1000
        if (cost_share.find("L")!=-1):
            limit = np.array(re.findall(r'L(\d+)',cost_share)).astype('int64')
            allowed = np.minimum(allowed,limit * avg_allowed_per_service)
            services = np.minimum(services,limit)
        #add allowed dollars
        clmnt_df['allowed'] = clmnt_df['allowed'] + allowed
        # use regular expression parsing to determine cost sharing
        special_coinsurance=0
        subj_to_ded = True
        subj_to_coin = True
        # does special coinsurance apply?
        if cost_share.find('%')!=-1:
            #special coinsurance applies, as well
            special_coinsurance  =np.array(re.findall(r'(\d+)%',cost_share)).astype('int64')
        if cost_share.find('ND')!=-1: subj_to_ded = False
        if cost_share.find('NC')!=-1: subj_to_coin = False
        if (cost_share.find('$')!=-1):
            #copay(s)
            subj_to_coin = False #default is not subject to  coinsurance unless special
            if cost_share.find('DED')!=-1: 
                subj_to_ded = True
            else:
                subj_to_ded = False
            if (cost_share.find('#')!=-1):
                # code for multiple copays
                # the code also works for a single copay, but much slower
                copays =  np.array(re.findall(r'\$(\d+)',cost_share)).astype('int64')
                nbr_times = np.array(re.findall(r'#(\d+)',cost_share)).astype('int64')
                pay_pattern = np.repeat(copays,np.append(nbr_times,limit))
                copay_df = pd.DataFrame({'services': services, 'avg_cost': avg_allowed_per_service})
                copay_cost = copay_df.apply(lambda x: apply_copays(int(x['services']),pay_pattern, x['avg_cost']), axis=1)
            else:
                # this is faster for base copay situation
                copay = np.array(re.findall(r'\$(\d+)',cost_share)).astype('int64')
                # next line ensures copay doesn't exceed the service cost
                copay = np.minimum(copay, avg_allowed_per_service)
                copay_cost = copay * services
            clmnt_df['copays'] = clmnt_df['copays'] + copay_cost
            clmnt_df['deductible'] = clmnt_df['deductible'] + (allowed-copay_cost) * subj_to_ded
            if special_coinsurance !=0:
                clmnt_df['coin'] = clmnt_df['coin'] + (allowed - copay_cost) * special_coinsurance
        elif special_coinsurance != 0:
            #just special coinsurance 
            clmnt_df['deductible'] = clmnt_df['deductible'] + allowed * subj_to_ded
            clmnt_df['coin'] = clmnt_df['coin'] + (allowed  * special_coinsurance) 
        elif (cost_share.find('F')!=-1):
            #covered in full, no accumulation
            clmnt_df['deductible'] = clmnt_df['deductible']
        else:
            #deductible and coinsurance applies
            clmnt_df['deductible'] = clmnt_df['deductible'] + allowed * subj_to_ded
            clmnt_df['coin'] = clmnt_df['coin'] + allowed * coinsurance * subj_to_coin
    
    clmnt_df['excess_ded'] = np.maximum(0,clmnt_df['deductible']-deductible)
    clmnt_df['coin']= np.divide(clmnt_df['excess_ded'],clmnt_df['deductible'], \
                                out=np.zeros_like(clmnt_df['excess_ded']), \
                                where=clmnt_df['deductible']!=0) * \
                      clmnt_df['coin']
    clmnt_df['deductible'] = np.minimum(clmnt_df['deductible'],deductible)
    clmnt_df['coin']= np.minimum(clmnt_df['coin'],moop - clmnt_df['deductible'] -clmnt_df['copays']) 
    clmnt_df['paid']= clmnt_df['allowed'] - clmnt_df['deductible'] - clmnt_df['coin'] - clmnt_df['copays']
    
    # run n scenarios
    n = int(750000/pop_size)
    #for i in range(n):
    accum_grp_allowed = np.zeros(n)
    accum_grp_paid = np.zeros(n)
    
    members = 0
    #iterate through population list
    for pop_group in population:
        clmnts = clmnt_df[(clmnt_df['age_group']==pop_group['agegroup']) & \
                          (clmnt_df['gender']==pop_group['gender'])]. \
                 sample(n=pop_group['count']*n, replace = True) 
        accum_grp_allowed = accum_grp_allowed + clmnts['allowed'].values. \
            reshape(-1,pop_group['count']).sum(1)
        accum_grp_paid = accum_grp_paid + clmnts['paid'].values. \
            reshape(-1,pop_group['count']).sum(1)
        members = members + pop_group['count']
    
    AV_list = np.divide(accum_grp_paid, accum_grp_allowed)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    allowed_PMPM = accum_grp_allowed / members
    ptile_allowed_PMPM = np.percentile(allowed_PMPM,[5,50,95])/12
    paid_PMPM = accum_grp_paid / members
    ptile_paid_PMPM = np.percentile(paid_PMPM,[5,50,95])/12
    ptile_AV = ptile_paid_PMPM / ptile_allowed_PMPM
    
    return ptile_AV,AV_list

### load datasets and global variables
# read in plan design
response = requests.get('https://github.com/morrowmike/actuarial-tools/blob/master/design_test.txt?raw=true')
design = json.loads(response.text)
# read in population
response = requests.get('https://github.com/morrowmike/actuarial-tools/blob/master/pop_test.txt?raw=true')
population = json.loads(response.text) 
# read in av model claimant data
response = requests.get('https://github.com/morrowmike/actuarial-tools/blob/master/AVSummarizedData.csv?raw=true')
clmnt_df = pd.read_csv(io.StringIO(response.decode('utf-8')),index_col=0)

### plan design form
Ded_input = dbc.FormGroup([
    dbc.Label("Individual Deductible",width = 8),
    dbc.Col(
        dbc.Input(type="text",id="input_ded", value=design['ind_ded']),
        width=4),],
    row=True,)
Coin_input = dbc.FormGroup([
    dbc.Label("Member Coinsurance",width = 8),
    dbc.Col(
        dbc.Input(type="text",id="input_coin", value=design['coin']),
        width=4),],
    row=True,)
Moop_input = dbc.FormGroup([
    dbc.Label("Maximum Out-of-Pocket",width = 8),
    dbc.Col(
        dbc.Input(type="text",id="input_moop", value=design['ind_oop']),
        width=4),],
    row=True,)
Overall_form = dbc.Form([Ded_input, Coin_input, Moop_input])

IP_input = dbc.FormGroup([
    dbc.Label("Inpatient Visits",width = 8),
    dbc.Col(dbc.Input(type="text",id="input_ip", value=design['IP_visit']),
            width=4),],
    row=True,)
OP_input = dbc.FormGroup([
    dbc.Label("Outpatient Surgery",width = 8),
    dbc.Col(dbc.Input(type="text",id="input_op", value=design['OP_surgery']),
            width=4),],
    row=True,)
ER_input = dbc.FormGroup([
    dbc.Label("ER Visits",width =8),
    dbc.Col(dbc.Input(type="text",id="input_er", value=design['ER_visit']),
            width=4),],
    row=True,)
Facility_form = dbc.Form([IP_input, OP_input, ER_input])

PCP_input = dbc.FormGroup([
    dbc.Label("Primary Care Visits",width = 8),
    dbc.Col(dbc.Input(type="text",id="input_pcp", value=design['PCP_visits']),
            width=4),],
    row=True,)
SPC_input = dbc.FormGroup([
    dbc.Label("Specialist Visits",width = 8),
    dbc.Col(dbc.Input(type="text",id="input_spc", value=design['Specialist_visits']),
            width=4),],
    row=True,)
PRV_input = dbc.FormGroup([
    dbc.Label("Preventive Visits",width = 8),
    dbc.Col(dbc.Input(type="text",id="input_prv", value=design['Prev_visits']),
            width=4),],
    row=True,)
CHI_input = dbc.FormGroup([
    dbc.Label("Chiropractice Visits",width = 8),
    dbc.Col(dbc.Input(type="text",id="input_chi", value=design['Chiropractic']),
            width=4),],
    row=True,)
Physician_form = dbc.Form([PCP_input, SPC_input, PRV_input, CHI_input])

GEN_input = dbc.FormGroup([
    dbc.Label("Generic Drugs",width = 8),
    dbc.Col(dbc.Input(type="text",id="input_gen", value=design['Rx_generic']),
            width=4),],
    row=True,)
BRD_input = dbc.FormGroup([
    dbc.Label("Brand Drugs",width = 8),
    dbc.Col(dbc.Input(type="text",id="input_brd", value=design['Rx_brand']),
            width=4),],
    row=True,)
Pharmacy_form = dbc.Form([GEN_input, BRD_input])

sub_button = dbc.Button("Update AV Calculation", color="primary", block = True,
                        id = "submit_button", n_clicks = 0)

size_slider = dcc.Slider(
    id= "size_slider",
    min = 200,
    max = 5000,
    value = 2000,
    marks = {
        250: {'label': '250 mbrs'},
        500: {'label': '500'},
        1000: {'label': '1,000'},
        1500: {'label': '1,500'},
        2000: {'label': '2,000'},
        3000: {'label': '3,000'},
        4000: {'label': '4,000'},
        5000: {'label': '5,000 mbrs'},
        },
    included = True)

### app layout
app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])
server = app.server
controls = dbc.Card(
    [
         Overall_form,
         Facility_form,
         Physician_form,
         Pharmacy_form,
         sub_button
     ],
         body=True,
    )

app.layout = dbc.Container([
    html.Div([html.H2('BST AV Calculator Prototype')]),
    html.Hr(),
    dbc.Row([
        dbc.Col(controls, 
                md=5),
        dbc.Col(
            children=dbc.Card(children=[
                dbc.CardHeader(html.H5(id='message')),
                dbc.CardBody(children =[
                    html.Div(id='interval_text'),
                    dcc.Graph(id='histogram')])
                ]),
            md=5),
        ],
        align='center',),
    html.Div(size_slider),
    html.Div(id='saved-data',style={'display':'none'})
    ],
    fluid=True)

@app.callback(Output('saved-data', 'children'),
              [Input('submit_button','n_clicks')],
              [State('input_ded', 'value'),State('input_coin', 'value'),
               State('input_moop', 'value'),State('input_ip', 'value'),
               State('input_op', 'value'),State('input_er', 'value'),
               State('input_pcp', 'value'),State('input_spc', 'value'),
               State('input_prv', 'value'),State('input_chi', 'value'),
               State('input_gen', 'value'),State('input_brd', 'value'),
               State('size_slider', 'value')
               ])
def evaluate_AV(n_clicks, input_ded, input_coin, input_moop,input_ip,
                  input_op, input_er, input_pcp, input_spc, input_prv,
                  input_chi, input_brd, input_gen, size_slider): 
    local_design = design
    local_design['ind_ded']= input_ded
    local_design['coin'] = input_coin
    local_design['ind_oop'] = input_moop
    local_design['IP_visit'] =input_ip
    local_design['OP_surgery'] = input_op
    local_design['PCP_visits'] = input_pcp
    local_design['Specialist_visits'] = input_spc
    local_design['Prev_visits'] = input_prv
    local_design['Chiropractic'] = input_chi
    local_design['Rx_generic'] = input_gen
    local_design['Rx_brand'] = input_brd
    local_population = population
    
    #scale local_population to desired size - standard size is 2040
    desired_size = size_slider
    scalar = desired_size / 2040
    for x in local_population:
        x['count'] = max(int(x['count'] * scalar),1)
    
    ptile_AV,AV_list = calculateAV(local_design,local_population,desired_size)
    result = {'percentiles':ptile_AV, 'AV_list':AV_list}
    return json.dumps(result, cls=NumpyEncoder)

@app.callback(Output('histogram','figure'),[Input('saved-data','children')])
def update_histogram(jsonified_result):
    result = json.loads(jsonified_result)
    AV_list = np.asarray(result['AV_list'])
    figure = go.Figure(data=[go.Histogram(x=AV_list)])
    figure.update_layout(
        xaxis_title="Actuarial Value",
        yaxis_title="Scenarios",
        title="Histogram of Forecast Simulations",
        template="plotly_dark")
    return figure

@app.callback(Output('message','children'),
             [Input('saved-data','children')])
def update_message(jsonified_result):
    result = json.loads(jsonified_result)
    ptile_AV = np.asarray(result['percentiles'])
    message = 'The median AV for this group is '+'{:.1%}'.format(ptile_AV[1])
    return message

@app.callback(Output('interval_text','children'),
             [Input('saved-data','children')])
def update_interval(jsonified_result):
    result = json.loads(jsonified_result)
    ptile_AV = np.asarray(result['percentiles'])
    message = 'The 90% AV prediction interval is between '+'{:.1%}'.format(ptile_AV[0]) \
              + ' and ' + '{:.1%}'.format(ptile_AV[2])
    return message

if __name__ == '__main__':
    app.run_server()
    
    
