# Hail on Manifold

## Initial Setup

### Manifold Platform

- Pick a workbench where you have compute environment permissions
- Create a new compute environment:
    - Click "Add Environment"
    - Enter a title
    - Choose the scipy jupyter notebook image.
    - Click "Add & Launch"
    - Wait for it to spin up, then open

### Terminal

Within the newly opend jupyter notebook, open a Terminal tab from the menu.

Download and run the Hail kernel installation script:

```bash
curl -fsSL https://raw.githubusercontent.com/hail-is/hail/main/user-scripts/manifold/install_hail_kernel.sh | bash
```

## Using Hail on the Manifold Platform

- Choose a suitable working directory and navigate to it on the left hand side panel:
  - `/home/jovyan/` - is the default user's home directory and opens by default. It is fine but it probably makes sense
    to use an explicitly chosen subdirectory.
  - `/home/jovyan/workbench/username/{project}` - will be preserved and shared across all environments in the workspace.
  - `/home/jovyan/local/{project}` - anything in `local` will be local to the current environment and not shared with anyone else.
  
- When navigated to the appropriate directory, open a new tab in the notebook. The new kernel might take a few seconds to appear
under Notebooks and Console
- When it appears (it will be called `Python (hail)`), open it 

### Example batches

You can now run Hail commands in the notebook:

#### Hello world
```python
import hailtop.batch as hb

b = hb.Batch('hello test')
j = b.new_job('hello task')
j.command('echo hello')
b.run()
```

#### File maker & updater

- First, create a file in the workbench at `~/workbench/world.txt` with the content "world"
- Now you can run the script below. It creates a temporary file with the content "hello" and 
then updates it with the content of the file in the workbench. Finally, it prints the contents of the file.


```python
import hailtop.batch as hb

b = hb.Batch('local file updater')

j1 = b.new_job('create_hello')
j1.command(f'echo "hello" > {j1.ofile}')

j2 = b.new_job('update_world')
j2.command(f'cat {j1.ofile} > {j2.ofile}; cat ~/workbench/world.txt >> {j2.ofile}')
b.write_output(j2.ofile, 'output.txt')

j3 = b.new_job('print_result')
j3.command(f'cat {j2.ofile}')

b.run()
```
