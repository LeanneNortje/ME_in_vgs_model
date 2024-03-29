# Using pretrained model checkpoints

If you want to use the model checkpoints, download the checkpoints given in the release and extract the checkpoints in the appropriate model folders.
<!---The following table gives the names in the releases that corresponds to the names in the ```model_metadata``` folder.-->

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Self-supervised AlexNet warmstart</th>
      <th>Self-supervised CPC warmstart</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">MattNet</td>
      <td><ul><li>- [ ] </li></ul></td>
      <td><ul><li>- [ ] </li></ul></td>
      <td>a</td>
    </tr>
    <tr>
      <td><ul><li>- [x] </li></ul></td>
      <td><ul><li>- [ ] </li></ul></td>
      <td>b</td>td
    </tr>
    <tr>
      <td><ul><li>- [ ] </li></ul></td>
      <td><ul><li>- [x] </li></ul></td>
      <td>c</td>
    </tr>
    <tr>
      <td><ul><li>- [x] </li></ul></td>
      <td><ul><li>- [x] </li></ul></td>
      <td>d</td>
    </tr>
    <tr>
      <td>MattNet using a hinge loss</td>td
      <td><ul><li>- [x] </li></ul></td>
      <td><ul><li>- [x] </li></ul></td>
      <td>c</td>
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
