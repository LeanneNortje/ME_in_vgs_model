# Using pretrained model checkpoints

If you want to use the model checkpoints, download the checkpoints given in the release and extract the checkpoints in the appropriate model folders.
<!---The following table gives the names in the releases that corresponds to the names in the ```model_metadata``` folder.-->

Model   | Self-supervised AlexNet warmstart    | Self-supervised CPC warmstart  | Name 
---    | ---   | ---   | ---    
MattNet | [x] | [x] | a

```bash
├── model_metadata
│   ├── <model_name>
│   │   ├── <model_instance>
│   │   │   ├── models
│   │   │   ├── args.pkl
│   │   │   ├── params.json
│   │   │   ├── training_metadata.json
```
