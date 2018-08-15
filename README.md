# Davos dataset:
## Input files:
- **user_cs_profile.csv**: CS usage profiles (SD, QESD, PR, HE, VSN, NONE) for each user (one per session)
- **user_cs_profile_clusters.xlsx**: k-means clustering results for 5 different random initializations, final cluster (based on majority vote), avg. (and std. dev.) for use of every CS per cluster, MANOVA results for comparison of mean CS use
- **final_clusters.csv**: Final clusters selected based on majority vote

- **rapport_evolution_full.txt**: Rapport evolution (thin slice rapport values) for all sessions

- **reco_feedback_full.txt**: Binary recommendation feedbacks (0: TopLink message rejected, 1: TopLink message accepted)
- **reco_feedback_partial.txt**: Recommendation feedbacks in which 0 (TopLink message rejected) has been split into finer categories. 3: rejected without any specified reason, 2: rejected with a specified reason, 4: positive feedback but TopLink message rejected, 5: session already over, 6: positive feedback but SARA didn't ask for TopLink message. 5 and 6 were removed from the analysis.
- **length_full.txt**: Length of each session in seconds
- **reco_sess_or_person.txt**: Type of recommendation (S: session, P: person)
- **reco_times_full**: Time stamps of recommendations

- **nlg_database.csv**: Template-based NLG database containing system natural language utterances for every combination of task and conversational strategy
- **transcripts/**: Folder containing text transcripts of users' interactions with SARA (each file is named according to the session id)

- **all_task_intentions_act_map_final.csv**: Task intentions to (reduced) task strategy map
- **user_cs/**: Folder containing user CS annotations

- **last_slice_length_full.txt**: Length of the last thin slice for every session

## Key output files:
- **clusters_full.pkl**: Dictionary with sessions as keys and clusters (0: P-Type, 1: I-Type) as values
- **id_to_f_full.pkl**: Dictionary with session (ids) as keys and cubic spline interpolation functions as values
- **train_data_full.pkl**: Dictionary of dictionaries. Contains training data for cluster-0, 1 and all sessions. Each dictionary has as keys: "user_cs_outp", "rapp_outp", "user_cs_inp_t-1", "user_cs_inp_t-2", "agent_cs_inp_t-0", "agent_cs_inp_t-1", "agent_cs_inp_t-2", "agent_intention_inp_t-0", "agent_intention_inp_t-1", "rapp_inp_t-1", "rapp_inp_t-2"

## Dataset annotations:
- **agent_cs/**: Folder containing 3 files per session (session-id):
  - **session-id_agent_cs.pkl**: Dictionary with turn numbers as keys and agent CS list (ASN, ACK, SD, QESD, PR, HE, VSN, NONE) as values
  - **session-id_agent_intention.pkl**: Dictionary with turn numbers as keys and agent task strategy list as values
  - **session-id_agent_timestamps.pkl**: Dictionary with turn numbers as keys and timestamp (end time of turn) (in seconds) as values

- **user_cs/pickle_files/**: Folder containing 2 files per session (session-id):
  - **session-id_user_cs.pkl**: Dictionary with turn numbers as keys and user CS list (SD, QESD, PR, HE, VSN, NONE) as values
  - **session-id_user_timestamps.pkl**: Dictionary with turn numbers as keys and timestamp (end time of turn) (in seconds) as values
