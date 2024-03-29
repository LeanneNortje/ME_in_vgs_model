# Using pretrained model checkpoints

If you want to use the model checkpoints, download the checkpoints given in the release and extract the checkpoints in the appropriate model folders.
<!---The following table gives the names in the releases that corresponds to the names in the ```model_metadata``` folder.-->

Model   | Self-supervised AlexNet warmstart    | Self-supervised CPC warmstart  | Name 
---    | ---   | ---   | ---    
MattNet | <ul><li>- [ ] </li></ul> | <ul><li>- [ ] </li></ul> | a
^^ | <ul><li>- [x] </li></ul> | <ul><li>- [ ] </li></ul> | a
^^ | <ul><li>- [ ] </li></ul> | <ul><li>- [x] </li></ul> | a
^^ | <ul><li>- [x] </li></ul> | <ul><li>- [x] </li></ul> | a

<table>
  <thead>
    <tr>
      <th colspan="2">1</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">1</td>
      <td rowspan="2">1</td>
      <td rowspan="2" colspan="2">2</td>
      <td>6</td>
    </tr>
    <tr>
      <td>7</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3</td>
      <td colspan="2">5</td>
    </tr>
  </tbody>
</table>

```bash
├── model_metadata
│   ├── <model_name>
│   │   ├── <model_instance>
│   │   │   ├── models
│   │   │   ├── args.pkl
│   │   │   ├── params.json
│   │   │   ├── training_metadata.json
```
