from fitparse import FitFile
import pandas as pd

def get_records (fit_file,record_type,amount=None):
    all_records = [] ## Defines an array for all messages in the file
    for count,message in enumerate(fit_file.get_messages (record_type)): ## iterates through all messages of a certain type
        if amount is not None and count >= amount: ## sets a limit in case the user wants a limit
            break
        ## here I define what I want to see, a message is structured in fields. Therfore I want to have an dictionary with all fields of a message
        field_content = {
            field.name : field.value
            for field in message if field.name and 'unknown' not in field.name
        }
        all_records.append (field_content)
    return all_records

def get_dataframe (fit_file,record_type,index=None):
    if index:
        df = pd.DataFrame(get_records(fit_file,record_type)).set_index(index)
    else:
        df = pd.DataFrame(get_records(fit_file,record_type))
    return df

def find_gaps (timestamps):
    time_gap = []
    for count,timestamp in enumerate(timestamps):
        if count > 0:
            timestampdiff = timestamp - timestamps[count-1] 
            if timestampdiff.total_seconds() > 1:
                time_gap.append ({
                    'start_time': timestamps[count-1],
                    'end_time': timestamp,
                    'gap': timestampdiff.total_seconds()
                })
    return time_gap

def get_cleaned_records (fit_file):
    records = get_dataframe (fit_file,'record','timestamp')
    print (find_gaps(records.index))
    records['cadence'] = records['cadence'].ffill(limit=10)
    records['heart_rate'] = records['heart_rate'].ffill(limit=3)
    records['power'] = records['power'].ffill(limit=1)
    return records

def get_baseline_timestamps(timestamps,baseline_minute_start,baseline_minute_end):
    ride_start = timestamps[0]
    baseline_start = ride_start + pd.Timedelta(minutes=baseline_minute_start)
    baseline_end = ride_start + pd.Timedelta(minutes=baseline_minute_end)
    return [baseline_start,baseline_end]
        
def get_baselines (parameter,dataFrame,baseline_start,baseline_end):
    baseline_timestamps = get_baseline_timestamps(sorted(dataFrame.index),baseline_start,baseline_end)
    baselines = []
    for p in parameter:
        baselines.append({
            'name': p,
            'value':dataFrame[p][baseline_timestamps[0]:baseline_timestamps[1]].mean()})
    return baselines

def add_delta_features (dataFrame,baselines):
    for b in baselines:
        dataFrame[f'delta_{b['name']}'] = dataFrame[f'{b['name']}'] - b['value'] 
    return dataFrame


activity = FitFile('22105359350_ACTIVITY.fit')

records = get_cleaned_records(activity)

baselines = get_baselines (['heart_rate','power'],records,5,15)

add_delta_features (records,baselines)
