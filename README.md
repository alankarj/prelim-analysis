# SARA Conference dataset (2017)
## Input files
- **user_cs_profile.csv**: CS usage profiles (SD, QESD, PR, HE, VSN, NONE) for each user (one per session)
- **user_cs_profile_clusters.xlsx**: k-means clustering results for 5 different random initializations, final cluster (based on majority vote), avg. (and std. dev.) for use of every CS per cluster, MANOVA results for comparison of mean CS use
- **final_clusters.csv**: Final clusters selected based on majority vote

- **rapport_evolution_full.txt**: Rapport evolution (thin slice rapport values) for all sessions

- **reco_feedback_full.txt**: Binary recommendation feedbacks (0: TopLink message rejected, 1: TopLink message accepted)
- **reco_feedback_partial.txt**: Recommendation feedbacks in which 0 (TopLink message rejected) has been split into finer categories. 3: rejected without any specified reason, 2: rejected with a specified reason, 4: positive feedback but TopLink message rejected, 5: session already over, 6: positive feedback but SARA didn't ask for TopLink message. 5 and 6 were removed from the analysis.
- **length_full.txt**: Length of each session in seconds
- **reco_sess_or_person.txt**: Type of recommendation (S: session, P: person)
- **reco_times_full**: Time stamps of recommendations

- **user_cs/**: Folder containing user CS annotations

- **last_slice_length_full.txt**: Length of the last thin slice for every session

## Key output files
- **clusters_full.pkl**: Dictionary with sessions as keys and clusters (0: P-Type, 1: I-Type) as values
- **id_to_f_full.pkl**: Dictionary with session (ids) as keys and cubic spline interpolation functions as values
- **train_data_full.pkl**: Dictionary of dictionaries. Contains training data for cluster-0, 1 and all sessions. Each dictionary has as keys: "user_cs_outp", "rapp_outp", "user_cs_inp_t-1", "user_cs_inp_t-2", "agent_cs_inp_t-0", "agent_cs_inp_t-1", "agent_cs_inp_t-2", "agent_intention_inp_t-0", "agent_intention_inp_t-1", "rapp_inp_t-1", "rapp_inp_t-2"
- **weights_re_<cluster_id>.t7**: Weights for rapport estimator.
- **weights_sr_user_<cluster_id>.t7**: Weights for the user social reasoner.
- **weights_sr_agent.t7**: Weights for the agent social reasoner.

## Dataset annotations
- **agent_cs/**: Folder containing 3 files per session (session-id):
  - **session-id_agent_cs.pkl**: Dictionary with turn numbers as keys and agent CS list (ASN, ACK, SD, QESD, PR, HE, VSN, NONE) as values
  - **session-id_agent_intention.pkl**: Dictionary with turn numbers as keys and agent task strategy list as values
  - **session-id_agent_timestamps.pkl**: Dictionary with turn numbers as keys and timestamp (end time of turn) (in seconds) as values

- **user_cs/pickle_files/**: Folder containing 2 files per session (session-id):
  - **session-id_user_cs.pkl**: Dictionary with turn numbers as keys and user CS list (SD, QESD, PR, HE, VSN, NONE) as values
  - **session-id_user_timestamps.pkl**: Dictionary with turn numbers as keys and timestamp (end time of turn) (in seconds) as values
  
## Training models
Modify the src/config.py file according to train/test condition, type of model, etc. and then run `python src/main.py`.

# References
Please cite the following paper if you found the code or the datasets in this or the social_user_simulator repositories useful.

A. Jain, F. Pecune, Y. Matsuyama and J. Cassell, [A user simulator architecture for socially-aware conversational agents](https://dl.acm.org/citation.cfm?id=3267916)

```
@inproceedings{jain2018user,
  title={A user simulator architecture for socially-aware conversational agents},
  author={Jain, Alankar and Pecune, Florian and Matsuyama, Yoichi and Cassell, Justine},
  booktitle={Proceedings of the 18th International Conference on Intelligent Virtual Agents},
  pages={133--140},
  year={2018},
  organization={ACM}
}
```

# Contact
Please reach out to me at alankarjain91@gmail.com in case of any questions or concerns.
