import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from datetime import datetime
import os
from groq import Groq # Import Groq library

# --- Configuration ---
DATA_FOLDER = os.path.dirname(os.path.abspath(__file__))
BACKEND_DATA_PATH = os.path.join(DATA_FOLDER, 'test_data_backend.csv')
SHAP_DATA_PATH = os.path.join(DATA_FOLDER, 'test_shap.csv')

# Configure Groq API Key:
# It's highly recommended to set this as an environment variable in production.
# For local testing, you can temporarily put your key directly here, but remove before committing to public repos.
GROQ_API_KEY = "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" # REPLACE WITH YOUR ACTUAL GROQ API KEY OR SET ENV VAR

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
            trial = {
                "id": str(row['Trial_ID']),
                "title": f"Trial for {row.get('Therapeutic Area', 'Unknown Area')} in {row.get('Study Status', 'Unknown Status')} - {row['Trial_ID']}",
                "sponsor": str(row.get('Sponsor', 'Unknown Sponsor')),
                "therapeuticArea": str(row.get('Therapeutic Area', 'Unknown Area')),
                "status": str(row.get('Study Status', 'Unknown')),
                "pts": float(round(row.get('PTS', 0.0), 1)),
                "enrollment": int(row.get('Enrollment', 0)),
                "countries": int(row.get('Country Count', 0)),
                "duration": int(row.get('Study Duration', 0)),
                "startYear": int(row.get('Start_Date_Year', 2020)),
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


# --- NEW: Groq AI Assistant Endpoint ---

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
                        "default": 70
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
                        "description": "Minimum PTS score for trials to be considered for sponsor ranking. Defaults to 80.",
                        "default": 80
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
                "id": t['id'],
                "sponsor": t['sponsor'],
                "pts": t['pts'],
                "enrollment": t.get('enrollment', 'N/A'), # Ensure enrollment is present
                "status": t['status']
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
                f"{s['sponsor']}: {s['averagePTS']:.1f}% Avg PTS, {s['totalTrials']} Trials"
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
                "id": t['id'],
                "sponsor": t['sponsor'],
                "therapeuticArea": t['therapeuticArea'],
                "pts": t['pts'],
                "enrollment": t.get('enrollment', 'N/A'), # Ensure enrollment is present
                "status": t['status'] # Ensure status is present
            })

        return {
            "type": "table",
            "title": f"Trials with '{endpoint_type}' as an Endpoint",
            # Frontend uses hardcoded headers, so this 'columns' array is primarily for semantic info or future dynamic rendering
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
    return {"error": "Tool not found."}


@app.route('/api/chat', methods=['POST'])
def chat_with_ai():
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        return jsonify({"success": False, "message": "Groq API key not configured. Please set GROQ_API_KEY environment variable or replace placeholder."}), 500

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
            model="llama-3.1-70b-versatile",
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=2048
        )

        tool_calls = chat_completion.choices[0].message.tool_calls
        if tool_calls:
            messages.append(chat_completion.choices[0].message)

            tool_outputs = [] # Collect outputs from all tool calls
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                print(f"LLM wants to call tool: {tool_name} with args: {tool_args}")

                tool_output = call_tool(tool_name, **tool_args)
                tool_outputs.append(tool_output) # Store the actual dict output

                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(tool_output)
                    }
                )
            
            final_completion = groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.1-70b-versatile",
                max_tokens=2048
            )
            llm_response_content = final_completion.choices[0].message.content
            
            # Prioritize structured tool output for frontend rendering
            if tool_outputs:
                # If multiple tools were called, you might choose to return the first one,
                # or combine them, depending on expected frontend behavior.
                # For now, let's return the first tool's structured output.
                return jsonify({"success": True, "data": tool_outputs[0]})
            else:
                return jsonify({"success": True, "data": {"type": "text", "message": llm_response_content}})

        else:
            llm_response_content = chat_completion.choices[0].message.content
            return jsonify({"success": True, "data": {"type": "text", "message": llm_response_content}})

    except Exception as e:
        app.logger.error(f"Error in Groq chat: {e}")
        return jsonify({"success": False, "message": f"An error occurred while processing your request: {str(e)}"}), 500

# --- How to run the Flask app for local development ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
