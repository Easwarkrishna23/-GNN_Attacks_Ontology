
The most impactful Attack is : **Evasion: Gradient (FGSM-like)**
Best post-defense (GCN): **Defense: Ontology + Retrain**
Best post-defense (GAT): **Defense: Ontology + Retrain**

=== FINAL TABLE (PRE-DEFENSE, GCN) ===
                       Attack  Accuracy       F1  ROC-AUC          Accuracy Drop  Perturbation Budget  NOTE
                     Baseline     0.802 0.792272 0.960516                    0.0                 0.00      
  Poisoning: Random Structure     0.482 0.477481 0.808305    0.32000000000000006              1500.00      
           Poisoning: Nettack     0.795 0.782785 0.958328   0.007000000000000006                64.00      
       Poisoning: Meta Attack     0.807 0.795950 0.966414 -0.0050000000000000044              2000.00      
           Evasion: Edge Flip     0.791 0.782021 0.956857    0.01100000000000001                80.00      
             Evasion: Feature     0.747 0.733418 0.932314    0.05500000000000005                12.00      
Evasion: Gradient (FGSM-like) **0.004** 0.002254 0.053342              **0.798**                 0.08 WORST

=== FINAL TABLE (POST-DEFENSE, GCN) ===
                              Attack  Accuracy       F1  ROC-AUC        Accuracy Drop  Perturbation Budget  NOTE
                            Baseline     0.802 0.792272 0.960516                  0.0                 0.00      
       Evasion: Gradient (FGSM-like) **0.004** 0.002254 0.053342            **0.798**                 0.08 WORST
Defense: Feature Smoothing + Retrain     0.887 0.895394 0.990561 -0.08499999999999996                 0.70      
          Defense: Pruning (top-k=5)     0.005 0.002796 0.048424                0.797                 0.50      
         Defense: Ontology + Retrain **0.937** 0.933337 0.993264           **-0.135**                 0.30  BEST
 Defense: Pruning+Ontology + Retrain     0.898 0.902613 0.991459 -0.09599999999999997                 0.60      

=== FINAL TABLE (PRE-DEFENSE, GAT) ===
                       Attack  Accuracy       F1  ROC-AUC         Accuracy Drop  Perturbation Budget  NOTE
                     Baseline     0.818 0.813076 0.972276                   0.0                 0.00      
  Poisoning: Random Structure     0.505 0.511968 0.826484   0.31299999999999994              1500.00      
           Poisoning: Nettack     0.811 0.805114 0.966351  0.006999999999999895               128.00      
       Poisoning: Meta Attack     0.815 0.806663 0.969134 0.0030000000000000027              4000.00      
           Evasion: Edge Flip     0.806 0.799278 0.969190    0.0119999999999999               120.00      
             Evasion: Feature     0.755 0.745691 0.939719   0.06299999999999994                12.00      
Evasion: Gradient (FGSM-like) **0.002** 0.001508 0.025493             **0.816**                 0.08 WORST

=== FINAL TABLE (POST-DEFENSE, GAT) ===
                              Attack  Accuracy       F1  ROC-AUC           Accuracy Drop  Perturbation Budget  NOTE
                            Baseline     0.818 0.813076 0.972276                     0.0                 0.00      
       Evasion: Gradient (FGSM-like) **0.002** 0.001508 0.025493               **0.816**                 0.08 WORST
Defense: Feature Smoothing + Retrain     0.927 0.922897 0.992488     -0.1090000000000001                 0.70      
          Defense: Pruning + Retrain     0.925 0.922727 0.993876     -0.1070000000000001                 0.50      
         Defense: Ontology + Retrain  **0.93** 0.927729 0.993639 **-0.1120000000000001**                 0.30  BEST
 Defense: Pruning+Ontology + Retrain     0.906 0.906542 0.992083    -0.08800000000000008                 0.60      
