import glob
import os
import random
from natsort import natsorted

def render_webpage(orig_dir, my_dir, foldit_dir, out_dir, randomize=False):
    my_preds = glob.glob(os.path.join(my_dir, '*.jpg'))
    foldit_files = glob.glob(os.path.join(foldit_dir, '*.jpg'))
    orig_files = glob.glob(os.path.join(orig_dir, '*.jpg'))

    my_preds = natsorted(my_preds)
    orig_files = natsorted(orig_files)
    foldit_files = natsorted(foldit_files)

    rand_swaps = [0]*len(orig_files)
    if randomize:
        rand_swaps = random.choices([0, 1], k=len(orig_files))
    
    both = [my_preds, foldit_files]

    with open(os.path.join(out_dir, 'index.html'), 'w') as h:
        h.write('<!DOCTYPE html>\n')
        h.write('<html>\n')
        h.write('<head><title>Comparison</title></head>\n')
        h.write('<body>\n')
        h.write(f'<!--{rand_swaps}-->\n')
        for i in range(len(orig_files)):
            base = my_preds[i].split('/')[-1].split('.')[0]
            h.write(f'<h3>{base} ({i+1})</h3>\n')
            h.write(f'''<table border="1" style="table-layout: fixed;">
      <tr>
        <td halign="center" style="word-wrap: break-word;" valign="top">
          <p>
            <a href="{orig_files[i]}">
              <img src="{orig_files[i]}" style="width:216px">
            </a><br>
            <p>Original</p>
          </p>
        </td>
        <td halign="center" style="word-wrap: break-word;" valign="top">
          <p>
            <a href="{both[rand_swaps[i]][i]}">
              <img src="{both[rand_swaps[i]][i]}" style="width:216px">
            </a><br>
            <p>No dense CRF</p>
          </p>
        </td>
        <td halign="center" style="word-wrap: break-word;" valign="top">
          <p>
            <a href="{both[1-rand_swaps[i]][i]}">
              <img src="{both[1-rand_swaps[i]][i]}" style="width:216px">
            </a><br>
            <p>Dense CRF</p>
          </p>
        </td>
      </tr>
    </table>
            ''')
        h.write('</body>\n')
        h.write('</html>')

if __name__=='__main__':
    folder = '/playpen/ridge-dtec/run/pascal/deeplab-mobilenet4-v2/ex_4_mollie'
    render_webpage(f'{folder}/original',
        f'{folder}/results-mine',
        f'/playpen/ridge-dtec/run/pascal/deeplab-mobilenet4-v2/ex_3_preston/results-mine',
        folder,
        randomize=False,
        )