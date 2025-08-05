import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS # Import CORS
import json
from datetime import datetime
import pickle
import shap
import os
from groq import Groq # Import Groq library

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
DATA_FOLDER = os.path.dirname(os.path.abspath(__file__))
BACKEND_DATA_PATH = os.path.join(DATA_FOLDER, 'test_data_backend.csv')
SHAP_DATA_PATH = os.path.join(DATA_FOLDER, 'test_shap.csv')
MODEL_PATH = os.path.join(DATA_FOLDER,"model.pkl")
# Configure Groq API Key:
# It's highly recommended to set this as an environment variable in production.
# For local testing, you can temporarily put your key directly here, but remove before committing to public repos.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE") 
app = Flask(__name__)
# IMPORTANT: For development/debugging in CodeSandbox, setting origins to "*"
# will bypass CORS issues. FOR PRODUCTION, ALWAYS SPECIFY YOUR FRONTEND DOMAINS!
CORS(app, origins=["https://n6dr7g.csb.app", "*"])

# --- JsonGenerator Class (reused from previous step) ---
class JsonGenerator:
    def __init__(self, df_backend, df_shap):
        self.df_backend = df_backend
        self.df_shap = df_shap
        self.base_pts_value = self.df_backend['PTS'].mean() if not self.df_backend.empty else 0.0

    def get_current_timestamp(self):
        return datetime.utcnow().isoformat(timespec='seconds') + 'Z'

    def generate_all_trials_json(self, page=1, limit=50, sort='PTS', order='desc', therapeutic_area=None, status=None, sponsor=None, search=None):
        df = self.df_backend.copy()

        # Apply filters
        if therapeutic_area:
            df = df[df['Therapeutic Area'].str.contains(therapeutic_area, case=False, na=False)]
        if status:
            df = df[df['Study Status'].str.contains(status, case=False, na=False)]
        if sponsor:
            df = df[df['Sponsor'].str.contains(sponsor, case=False, na=False)]
        if search:
            df = df[df['Trial_ID'].str.contains(search, case=False, na=False) |
                    df['Study Title Length'].astype(str).str.contains(search, case=False, na=False)]

        # Sort data
        if sort not in df.columns:
            if sort.lower() == 'pts':
                sort_col = 'PTS'
            else:
                sort_col = 'Trial_ID'
        else:
            sort_col = sort

        df_sorted = df.sort_values(by=sort_col, ascending=(order == 'asc'))

        # Pagination
        start_index = (page - 1) * limit
        end_index = start_index + limit
        paginated_df = df_sorted.iloc[start_index:end_index]

        trials_data = []
        for _, row in paginated_df.iterrows():
            # Safely get integer values, converting NaN to 0 or default if present
            enrollment_val = row.get('Enrollment')
            countries_val = row.get('Country Count')
            duration_val = row.get('Study Duration')
            start_year_val = row.get('Start_Date_Year')

            trial = {
                "id": str(row['Trial_ID']),
                "title": f"Trial for {row.get('Therapeutic Area', 'Unknown Area')} in {row.get('Study Status', 'Unknown Status')} - {row['Trial_ID']}",
                "sponsor": str(row.get('Sponsor', 'Unknown Sponsor')),
                "therapeuticArea": str(row.get('Therapeutic Area', 'Unknown Area')),
                "status": str(row.get('Study Status', 'Unknown')),
                "pts": float(round(row.get('PTS', 0.0), 1)),
                # Fixed NaN to int conversion here
                "enrollment": int(enrollment_val) if pd.notna(enrollment_val) else 0,
                "countries": int(countries_val) if pd.notna(countries_val) else 0,
                "duration": int(duration_val) if pd.notna(duration_val) else 0,
                "startYear": int(start_year_val) if pd.notna(start_year_val) else 2020,
                "primaryCountry": "United States", # Placeholder
                "hasOS": bool(row.get('Presence of \'Overall Survival\' (OS)', False)),
                "hasPFS": bool(row.get('Presence of \'Progression Free Survival\' (PFS)', False)),
                "hasORR": bool(row.get('Presence of \'Objective Response Rate\' (ORR)', False)),
                "age": str(row.get('Age', 'All')),
                "sex": str(row.get('Sex', 'All')),
                "primaryEndpoint": "Overall Survival" if row.get('Presence of \'Overall Survival\' (OS)', False) else \
                                   ("Progression Free Survival" if row.get('Presence of \'Progression Free Survival\' (PFS)', False) else "Other"),
                "phase": "Phase 3", # Placeholder
                "interventionType": "Drug", # Placeholder
                "studyDesign": str(row.get('Intervention Model', 'Unknown')),
                "createdAt": self.get_current_timestamp(),
                "updatedAt": self.get_current_timestamp()
            }
            trials_data.append(trial)

        response = {
            "success": True,
            "data": trials_data,
            "total": int(len(df_sorted)),
            "page": int(page),
            "limit": int(limit)
        }
        return response

    def generate_shap_data_json(self, trial_id):
        trial_row = self.df_backend[self.df_backend['Trial_ID'] == trial_id]
        if trial_row.empty:
            return {"success": False, "message": f"Trial ID {trial_id} not found."}
        trial_row = trial_row.iloc[0]

        shap_row = self.df_shap[self.df_shap['Trial_ID'] == trial_id]
        if shap_row.empty:
            return {"success": False, "message": f"SHAP data for Trial ID {trial_id} not found."}
        shap_row = shap_row.iloc[0]

        features_list = []
        shap_cols = [col for col in shap_row.index if col.startswith(('num__', 'cat__'))]

        shap_values_temp = {}
        for col in shap_cols:
            original_feature_name = col.replace('num__', '').replace('cat__', '')
            shap_values_temp[original_feature_name] = shap_row[col]

        sorted_features = sorted(shap_values_temp.items(), key=lambda item: abs(item[1]), reverse=True)[:5]

        for feature_name, shap_value in sorted_features:
            feature_value = None
            for col_backend in trial_row.index:
                if feature_name.lower().replace('_', '').replace(' ', '') == col_backend.lower().replace('_', '').replace(' ', ''):
                    feature_value = trial_row[col_backend]
                    break

            description = f"Impact of {feature_name} on PTS prediction."
            if pd.notna(feature_value):
                description = f"Impact of {feature_name} (Value: {feature_value}) on PTS prediction."
            else:
                feature_value = "N/A"

            features_list.append({
                "feature": str(feature_name),
                "shapValue": float(round(shap_value, 2)),
                "featureValue": str(feature_value),
                "description": str(description)
            })

        response = {
            "success": True,
            "data": {
                "trialId": str(trial_id),
                "baseValue": float(round(self.base_pts_value, 2)),
                "predictedPTS": float(round(trial_row['PTS'], 2)),
                "features": features_list,
                "riskFactors": [
                    {"factor": "Data incompleteness", "impact": "Low", "recommendation": "Verify missing values."},
                    {"factor": "Small sample size", "impact": "Medium", "recommendation": "Consider larger enrollment."}
                ],
                "confidenceInterval": {
                    "lower": float(round(trial_row['PTS'] - 5, 2)),
                    "upper": float(round(trial_row['PTS'] + 5, 2))
                },
                "generatedAt": self.get_current_timestamp()
            }
        }
        return response

    def generate_analytics_json(self):
        total_trials = int(len(self.df_backend))
        average_pts = float(round(self.df_backend['PTS'].mean(), 2))
        high_risk_trials = int(self.df_backend[self.df_backend['PTS'] < 50].shape[0])
        low_risk_trials = int(self.df_backend[self.df_backend['PTS'] >= 70].shape[0])
        completed_trials = int(self.df_backend[self.df_backend['Study Status'] == 'COMPLETED'].shape[0])

        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        labels = [f"{i}-{i+10}" for i in bins[:-1]]
        pts_ranges = pd.cut(self.df_backend['PTS'], bins=bins, labels=labels, right=False, include_lowest=True)
        pts_distribution_counts = pts_ranges.value_counts().sort_index()
        pts_distribution_list = [{"range": str(r), "count": int(c)} for r, c in pts_distribution_counts.items()]

        self.df_backend['Sponsor'] = self.df_backend['Sponsor'].fillna('Unknown')
        sponsor_performance = self.df_backend.groupby('Sponsor').agg(
            totalTrials=('Trial_ID', 'count'),
            averagePTS=('PTS', 'mean'),
            successRate=('Sponsor_Success_Rate', 'mean')
        ).reset_index()
        sponsor_performance_list = sponsor_performance.apply(
            lambda row: {
                "sponsor": str(row['Sponsor']),
                "totalTrials": int(row['totalTrials']),
                "averagePTS": float(round(row['averagePTS'], 2)),
                "successRate": float(round(row['successRate'] * 100, 2)) if pd.notna(row['successRate']) else 0.0
            }, axis=1
        ).tolist()

        self.df_backend['Therapeutic Area'] = self.df_backend['Therapeutic Area'].fillna('Unknown')
        therapeutic_area_breakdown = self.df_backend.groupby('Therapeutic Area').agg(
            count=('Trial_ID', 'count'),
            averagePTS=('PTS', 'mean')
        ).reset_index()
        therapeutic_area_breakdown_list = therapeutic_area_breakdown.apply(
            lambda row: {
                "therapeuticArea": str(row['Therapeutic Area']),
                "count": int(row['count']),
                "averagePTS": float(round(row['averagePTS'], 2))
            }, axis=1
        ).tolist()

        self.df_backend['Start_Date_Year'] = self.df_backend['Start_Date_Year'].fillna(0).astype(int)
        pts_over_time = self.df_backend.groupby('Start_Date_Year')['PTS'].mean().reset_index()
        pts_over_time_list = pts_over_time.apply(
            lambda row: {
                "year": int(row['Start_Date_Year']),
                "avgPTS": float(round(row['PTS'], 2))
            }, axis=1
        ).tolist()

        response = {
            "success": True,
            "data": {
                "summary": {
                    "totalTrials": total_trials,
                    "averagePTS": average_pts,
                    "highRiskTrials": high_risk_trials,
                    "lowRiskTrials": low_risk_trials,
                    "completedTrials": completed_trials
                },
                "ptsDistribution": pts_distribution_list,
                "sponsorPerformance": sponsor_performance_list,
                "therapeuticAreaBreakdown": therapeutic_area_breakdown_list,
                "trends": {
                    "ptsOverTime": pts_over_time_list
                },
                "generatedAt": self.get_current_timestamp()
            }
        }
        return response

# --- Data Loading (Global for the app instance) ---
df_backend = pd.DataFrame()
df_shap = pd.DataFrame()

with open(MODEL_PATH,'rb') as model_file:
    model = pickle.load(model_file)

json_gen = None

def load_data():
    global df_backend, df_shap, json_gen
    try:
        df_backend = pd.read_csv(BACKEND_DATA_PATH)
        df_shap = pd.read_csv(SHAP_DATA_PATH)
        json_gen = JsonGenerator(df_backend, df_shap)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: Data file not found: {e}. Please ensure CSVs are in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")

# Load data when the application starts
load_data()


# --- API Endpoints (Existing) ---

@app.route('/api/trials', methods=['GET'])
def get_all_trials():
    if json_gen is None or df_backend.empty:
        return jsonify({"success": False, "message": "API server not ready: Data not loaded."}), 503

    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    sort = request.args.get('sort', 'pts', type=str)
    order = request.args.get('order', 'desc', type=str)
    therapeutic_area = request.args.get('filter[therapeuticArea]', type=str)
    status = request.args.get('filter[status]', type=str)
    sponsor = request.args.get('filter[sponsor]', type=str)
    search = request.args.get('search', type=str)

    try:
        response_data = json_gen.generate_all_trials_json(
            page=page,
            limit=limit,
            sort=sort,
            order=order,
            therapeutic_area=therapeutic_area,
            status=status,
            sponsor=sponsor,
            search=search
        )
        return jsonify(response_data)
    except Exception as e:
        app.logger.error(f"Error in /api/trials: {e}")
        return jsonify({"success": False, "message": "An internal server error occurred."}), 500

@app.route('/api/trials/<trialId>/shap', methods=['GET'])
def get_shap_data_for_trial(trialId):
    if json_gen is None or df_backend.empty or df_shap.empty:
        return jsonify({"success": False, "message": "API server not ready: Data not loaded."}), 503

    try:
        response_data = json_gen.generate_shap_data_json(trialId)
        if not response_data.get("success"):
            return jsonify(response_data), 404
        return jsonify(response_data)
    except Exception as e:
        app.logger.error(f"Error in /api/trials/<trialId>/shap for trial {trialId}: {e}")
        return jsonify({"success": False, "message": "An internal server error occurred."}), 500

@app.route('/api/trials/analytics', methods=['GET'])
def get_analytics():
    if json_gen is None or df_backend.empty:
        return jsonify({"success": False, "message": "API server not ready: Data not loaded."}), 503

    try:
        response_data = json_gen.generate_analytics_json()
        return jsonify(response_data)
    except Exception as e:
        app.logger.error(f"Error in /api/trials/analytics: {e}")
        return jsonify({"success": False, "message": "An internal server error occurred."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    if json_gen is not None and not df_backend.empty:
        return jsonify({"status": "ok", "message": "API is running and data is loaded."}), 200
    else:
        return jsonify({"status": "degraded", "message": "API is running, but data not fully loaded."}), 503

@app.route('/status', methods=['GET'])
def status_check():
    return jsonify({"message": "Flask server is running!"}), 200


# --- Groq AI Assistant Tools and Client ---

def predict_pts(trial_row):
  print(trial_row[['Allocation','Masking']])
  preprocessor = model.named_steps['preprocessor']
  classifier = model.named_steps['classifier']
  test_df_transformed = preprocessor.transform(trial_row)
  preds = classifier.predict_proba(test_df_transformed)[:,1]
  return preds[0] * 100

def what_if_scenario_tool(trial_id, changes):
    global df_backend
    trial_row = df_backend[df_backend['Trial_ID'] == trial_id].copy()
    old_pts = trial_row['PTS'].values[0]
    send_changes = []
    for change_col , change_val in changes.items():
        send_changes.append({
            "column":change_col,
            "old_value":old_pts,
            "new_value":change_val
        })
        trial_row[change_col] = change_val
    new_pts = predict_pts(trial_row)
    if len(trial_row) == 0:
        return {
            trial_id:False
        }
    return {
        "trial_id":trial_id,
        "new_pts":new_pts,
        "old_pts":old_pts,
        "changes":send_changes
    }

# Initialize Groq client globally once
groq_client = Groq(api_key=GROQ_API_KEY)

# Define the tools available to the Groq LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_high_pts_oncology_trials",
            "description": "Retrieves clinical trials in the Oncology therapeutic area with high PTS scores. Filters by minimum PTS and limits results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_pts": {
                        "type": "integer",
                        "description": "Minimum PTS score to filter by.",
                        "default": 60
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of trials to return.",
                        "default": 5
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_sponsors_by_pts",
            "description": "Retrieves the top sponsors based on their average PTS scores or number of trials exceeding a certain PTS score.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_pts": {
                        "type": "integer",
                        "description": "Minimum PTS score for trials to be considered for sponsor ranking. Defaults to 60.",
                        "default": 60
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of top sponsors to return.",
                        "default": 5
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_trials_by_endpoint",
            "description": "Lists trials that use a specific primary endpoint type (e.g., Overall Survival, Progression Free Survival, Objective Response Rate).",
            "parameters": {
                "type": "object",
                "properties": {
                    "endpoint_type": {
                        "type": "string",
                        "enum": ["Overall Survival", "Progression Free Survival", "Objective Response Rate"],
                        "description": "The type of primary endpoint to filter trials by (OS, PFS, or ORR)."
                    }
                },
                "required": ["endpoint_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_global_failure_features",
            "description": "Provides insights into top features generally contributing to trial failures based on pre-analyzed global data. This tool does not take specific trial IDs.",
            "parameters": {
                "type": "object",
                "properties": {}, # No parameters for this global insight
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "what_if_scenario_tool",
            "description": "Facilitates what-if scenarios analysis by allowing users to modify key trial attributes such as enrollment, masking, intervention model, allocation, and primary purpose to see their impact on trial success probability.\
             It Can take Trial ID or NCT ID. It calculates and returns new and old trial success probability , where new success probability is for what if scenario, PTS stands for Probability of Trial Success",
            "parameters": {
            "type": "object",
            "properties": {
                "trial_id": {
                "type": "string",
                "description": "Unique identifier for the clinical trial to analyze. can also be reffered as NCTID or Trial_ID column"
                },
                "changes": {
                "type": "object",
                "description": "A dictionary of column-value pairs representing the attributes and their new hypothetical values for the trial. Valid columns include enrollment, masking, intervention model, allocation, and primary purpose.",
                "properties": {
                    "Enrollment": {
                    "type": "integer",
                    "description": "Hypothetical enrollment count."
                    },
                    "Masking": {
                    "type": "string",
                    "description": " Hypothetical masking type, will be one of these values if not then map to most appropriate: 'NONE', 'DOUBLE (PARTICIPANT, INVESTIGATOR)', 'QUADRUPLE (PARTICIPANT, CARE_PROVIDER, INVESTIGATOR, OUTCOMES_ASSESSOR)',\
                                    'TRIPLE (PARTICIPANT, INVESTIGATOR, OUTCOMES_ASSESSOR)','SINGLE (OUTCOMES_ASSESSOR)',\
                                    'TRIPLE (PARTICIPANT, CARE_PROVIDER, INVESTIGATOR)',\
                                    'DOUBLE (PARTICIPANT, CARE_PROVIDER)', 'SINGLE (PARTICIPANT)'"
                    },
                    "Intervention Model": {
                    "type": "string",
                    "description": "Hypothetical intervention model, will be one of these values if not then map to most appropriate 'PARALLEL', 'FACTORIAL', 'SEQUENTIAL', 'CROSSOVER', 'SINGLE_GROUP'."
                    },
                    "Allocation": {
                    "type": "string",
                    "description": "Hypothetical allocation method, will be one of these values if not then map to most appropriate : 'RANDOMIZED', 'NON_RANDOMIZED'."
                    },
                    "Primary Purpose": {
                    "type": "string",
                    "description": "Hypothetical primary purpose, will be one of these values if not then map to most appropriate :  'TREATMENT', 'DIAGNOSTIC'."
                    }
                },
                "additionalProperties": False
                }
      },
      "required": ["trial_id", "changes"]
    }
  }
}
]

# Tool Dispatcher: Maps tool names to functions
def call_tool(tool_name, **kwargs):
    if tool_name == "get_high_pts_oncology_trials":
        all_trials_data = json_gen.generate_all_trials_json(
            therapeutic_area="Oncology",
            limit=len(json_gen.df_backend), # Fetch all relevant data to filter client-side
            sort='PTS',
            order='desc'
        )
        min_pts = kwargs.get('min_pts', 70)
        limit = kwargs.get('limit', 5)

        filtered_trials = [
            t for t in all_trials_data['data'] if t['pts'] >= min_pts
        ]
        # Sort and limit after client-side filtering
        filtered_trials = sorted(filtered_trials, key=lambda x: x['pts'], reverse=True)[:limit]

        # Structure data to match frontend's hardcoded table rendering
        formatted_data = []
        for t in filtered_trials:
            formatted_data.append({
                "id": t.get('id', 'N/A'),
                "sponsor": t.get('sponsor', 'N/A'),
                "pts": t.get('pts', 0.0), # Ensure pts is a float
                "enrollment": t.get('enrollment', 'N/A'),
                "status": t.get('status', 'N/A')
            })

        return {
            "type": "table",
            "title": f"Top Oncology Trials with PTS >= {min_pts}%",
            # Frontend uses hardcoded headers, so this 'columns' array is primarily for semantic info or future dynamic rendering
            "columns": ["Trial ID", "Sponsor", "PTS", "Enrollment", "Status"],
            "data": formatted_data
        }

    elif tool_name == "get_top_sponsors_by_pts":
        analytics_data = json_gen.generate_analytics_json()
        min_pts = kwargs.get('min_pts', 80)
        limit = kwargs.get('limit', 5)

        top_sponsors = [
            s for s in analytics_data['data']['sponsorPerformance'] if s['averagePTS'] >= min_pts
        ]
        top_sponsors = sorted(top_sponsors, key=lambda x: x['averagePTS'], reverse=True)[:limit]

        return {
            "type": "list",
            "title": f"Sponsors with Average PTS >= {min_pts}% (Top {limit})",
            "data": [
                f"{s.get('sponsor', 'N/A')}: {s.get('averagePTS', 0.0):.1f}% Avg PTS, {s.get('totalTrials', 0)} Trials"
                for s in top_sponsors
            ]
        }

    elif tool_name == "get_trials_by_endpoint":
        endpoint_type = kwargs.get('endpoint_type')
        if not endpoint_type:
            return {"error": "Endpoint type is required."}

        # Use generate_all_trials_json to get data, fetch all if necessary
        all_trials_data = json_gen.generate_all_trials_json(limit=len(json_gen.df_backend))

        filtered_trials = []
        for trial in all_trials_data['data']:
            if (endpoint_type == "Overall Survival" and trial.get('hasOS', False)) or \
               (endpoint_type == "Progression Free Survival" and trial.get('hasPFS', False)) or \
               (endpoint_type == "Objective Response Rate" and trial.get('hasORR', False)):
                filtered_trials.append(trial)
        
        # Structure data to match frontend's hardcoded table rendering
        formatted_data = []
        for t in filtered_trials:
            formatted_data.append({
                "id": t.get('id', 'N/A'),
                "sponsor": t.get('sponsor', 'N/A'),
                "therapeuticArea": t.get('therapeuticArea', 'N/A'),
                "pts": t.get('pts', 0.0), # Ensure pts is a float
                "enrollment": t.get('enrollment', 'N/A'),
                "status": t.get('status', 'N/A')
            })

        return {
            "type": "table",
            "title": f"Trials with '{endpoint_type}' as an Endpoint",
            "columns": ["Trial ID", "Sponsor", "Therapeutic Area", "PTS", "Enrollment", "Status"],
            "data": formatted_data
        }

    elif tool_name == "get_global_failure_features":
        return {
            "type": "features",
            "title": "Top 5 Features Generally Contributing to Trial Failures",
            "features": [
                {"name": "Inadequate Enrollment Size", "impact": "23%", "description": "Trials with lower enrollment often face statistical power issues."},
                {"name": "Extended Study Duration", "impact": "18%", "description": "Overly long studies can lead to increased costs and dropouts."},
                {"name": "Lack of Biomarker Strategy", "impact": "15%", "description": "Without clear patient selection, drug efficacy might be diluted."},
                {"name": "Single-Country Studies", "impact": "12%", "description": "Limited geographic diversity can impact generalizability of results."},
                {"name": "Unclear Primary Endpoints", "impact": "10%", "description": "Ambiguous endpoint definition makes trial success difficult to measure."}
            ]
        }

    elif tool_name == "what_if_scenario_tool":
        what_if_prediction = what_if_scenario_tool(**kwargs)
        if what_if_prediction['trial_id'] == False:
            return {
                "error":"Trial ID Not Found"
            }
        return {
            "type":"whatif",
            "title":f"What If Scenario For {what_if_prediction['trial_id']}",
            "data": what_if_prediction
        }
    return {"error": "Tool not found."}


@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    # Check if API key is configured
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE" or not GROQ_API_KEY:
        return jsonify({"success": False, "message": "Groq API key not configured. Please set GROQ_API_KEY environment variable or replace placeholder."}), 500

    # Check if data is loaded
    if json_gen is None or df_backend.empty:
        return jsonify({"success": False, "message": "API server not ready: Data not loaded."}), 503

    data = request.get_json()
    user_message_content = data.get('message')

    if not user_message_content:
        return jsonify({"success": False, "message": "No message provided."}), 400

    messages = [
        {
            "role": "system",
            "content": "You are a clinical trials analysis assistant. Provide insights based on the provided trial data context. Use the available tools to answer specific data queries. If a query requires data not covered by tools, respond gracefully indicating so."
        },
        {
            "role": "user",
            "content": user_message_content
        }
    ]

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile", # Changed model here
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2048
        )

        tool_calls = chat_completion.choices[0].message.tool_calls
        if tool_calls:
            messages.append(chat_completion.choices[0].message)

            tool_outputs = [] # Collect outputs from all tool calls
            llm_explanation = "" # To store the LLM's final text explanation

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = {}
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    # Robustly handle malformed arguments from LLM
                    app.logger.error(f"LLM generated malformed JSON for tool {tool_name}: {tool_call.function.arguments}. Error: {e}")
                    # Provide a fallback message to the user
                    return jsonify({"success": False, "message": "The AI encountered an issue processing its internal thoughts. Please try rephrasing your query."}), 500


                tool_output = call_tool(tool_name, **tool_args)
                
                # Robustly check tool_output structure
                if isinstance(tool_output, dict) and 'type' in tool_output:
                    tool_outputs.append(tool_output) # Store the actual dict output
                else:
                    app.logger.error(f"Tool '{tool_name}' returned unexpected data type or missing 'type' key: {tool_output}")
                    # If tool output is bad, don't use it, and let LLM explain or fall back
                    llm_explanation = "The tool encountered an issue processing the data for your request."
                    # No structured data to return, proceed to get LLM's explanation if any
                    continue # Skip appending this bad tool output to messages for LLM

                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(tool_output) # Tool output must be a string
                    }
                )
            
            # Get final response from LLM after tool execution
            final_completion = groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant", # Changed model here
                max_tokens=2048
            )
            llm_response_content = final_completion.choices[0].message.content
            if tool_outputs:
                # If multiple tools were called, you might choose to return the first one,
                # or combine them, depending on expected frontend behavior.
                # For now, let's return the first tool's structured output.
                return jsonify({"success": True, "data": {**tool_outputs[0], "message":llm_response_content}})
            elif llm_explanation:
                # If a tool failed but we have an explanation, provide that
                 return jsonify({"success": True, "data": {"type": "text", "message": llm_explanation}})
            else:
                # Fallback to direct LLM response if no structured output was valid
                return jsonify({"success": True, "data": {"type": "text", "message": llm_response_content}})

        else:
            # No tool call, direct response from LLM
            llm_response_content = chat_completion.choices[0].message.content
            return jsonify({"success": True, "data": {"type": "text", "message": llm_response_content}})

    except Exception as e:
        app.logger.error(f"Error in Groq chat: {e}")
        return jsonify({"success": False, "message": f"An error occurred while processing your request: {str(e)}"}), 500

# --- How to run the Flask app for local development ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
