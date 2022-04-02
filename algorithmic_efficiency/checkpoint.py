import dill
from flax.training import checkpoints
import os
from absl import logging
from absl import flags
from algorithmic_efficiency import spec
from typing import Optional

# Note, although this should work for either pytorch or jax. It uses the flax 
# checkpointing module, and therefore that would need to be added to the setup.cfg
# in case someone doesn't have flax installed

FLAGS = flags.FLAGS

class Checkpointer:
    def __init__(self, ckpt_dir: str, workload: Optional[spec.Workload] = None, 
        num_train_samples: Optional[int] = None, ckpt_step: str = 'eval', 
        ckpt_freq: int=1, max_per_run: int=20, save_base_model=True, ) -> None:

        self.ckpt_dir = ckpt_dir
        self.ckpt_step = ckpt_step
        self.ckpt_freq = ckpt_freq
        
        self.max_per_run = max_per_run
        self.num_run_checkpoints = 0
        self.samples_sum = 0
        self.current_run = 0
        self.step = 0 # Keeps track of the current eval, epoch or step

        if workload is not None:
            self.num_train_samples = workload.num_train_examples
            if save_base_model:
                self.save_base_model(workload)
        elif num_train_samples is not None:
            self.num_train_samples = num_train_samples
        elif self.ckpt_step == 'epoch':
            logging.warn('Checkpoint epoch step can not be used unless workload \
                or num_train_samples is provided when instantiating Checkpointer')

        return

    def save_base_model(self, workload):
        '''Pickles a the model class object so that it can be used later to
        load the checkpoints.

        Args:
            save_path: Path to the directory in which to save the model
            model: An instance of the model class capable of loading the 
                model state and parameters
        '''
        get_model = getattr(workload, 'get_model_class', None)
        if callable(get_model):
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir)
            with open(os.path.join(self.ckpt_dir, 'model.pkl'), 'wb') as f:
                dill.dump(get_model(), f)
        else:
            logging.warn('Cannot save model instance. Workload does not have a get_model_class function.')

        return

    def save_checkpoint(self, params, model_state, step, sub_folder='checkpoints',
        prefix='checkpoint', keep=1, overwrite=False, keep_every_n_steps=1):
        '''Save a checkpoint of the model.

        Attempts to be pre-emption safe by writing to temporary before
        a final rename and cleanup of past files.

        Args:
            params: The model parameters. Usually a dictionary
            model_state: The model state. Usually a dictionary
            step: int or float: training step number or other metric number.
            prefix: str: checkpoint file name prefix.
            keep: number of past checkpoint files to keep.
            overwrite: overwrite existing checkpoint files if a checkpoint
            at the current or a later step already exits (default: False).
            keep_every_n_steps: if defined, keep every checkpoints every n steps (in
            addition to keeping the last 'keep' checkpoints).
        Returns:
            Filename of saved checkpoint.
        '''
        save_path = os.path.join(self.ckpt_dir, sub_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        param_dict = None
        try:
            param_dict = dict(params)
        except:
            to_dict = getattr(params, "state_dict", None)
            if callable(to_dict):
                param_dict = to_dict()
            else:
                logging.warn('Could not convert model params to a dict. Checkpoint not saved')
                return

        ckpt = {'params': param_dict, 'model_state': model_state}

        ckpt_path = checkpoints.save_checkpoint(
            ckpt_dir=save_path, 
            target=ckpt, 
            step=step, 
            prefix=prefix, 
            keep=keep, 
            overwrite=overwrite, 
            keep_every_n_steps=keep_every_n_steps
            )

        return ckpt_path

    def check_and_save(self, model_params, model_state, 
        current_step: int, current_eval: int, current_batch_size: int, 
        run_idx: int = 0, final=False):
        '''Calls save_checkpoint if it is time to save a new checkpoint.

        Args:
            params: The model parameters. Usually a dictionary
            model_state: The model state. Usually a dictionary
            current_step: The current training step
            current_eval: The number of evaluations that have been performed
            current_batch_size: The batch size of the previous step
            run_idx: The index of the current run/tuning trial
            final: If true, saves a checkpoint with the prefix final
        '''
        if run_idx > self.current_run:
            self.num_run_checkpoints = 0
            self.samples_sum = 0
            self.current_run = run_idx
            self.step = 0

        if self.num_run_checkpoints > self.max_per_run:
            return
        
        if self.ckpt_step == 'eval': 
            self.step = current_eval
        elif self.ckpt_step == 'step': 
            self.step = current_step
        elif self.ckpt_step == 'epoch':
            # In case in the future batch size is allowed to vary during training
            self.samples_sum += current_batch_size
            if self.num_train_samples < self.samples_sum:
                self.step += 1
                self.samples_sum = 0
        else:
            logging.warn('Do not recognize checkpoint step type.\
                Should be eval, epoch or step. Checkpoints will not be saved')
            return

        if ((self.step % self.ckpt_freq == 0 and 
            self.num_run_checkpoints != int(self.step/self.ckpt_freq))
            or final):
            if final:
                prefix = 'final_tune' + str(run_idx) + '_' + self.ckpt_step + '_'
            else:
                prefix = 'tune' + str(run_idx) + '_' + self.ckpt_step + '_'
            
            self.save_checkpoint(
                params=model_params,
                model_state=model_state,
                step=self.step,
                prefix=prefix,
            )
            self.num_run_checkpoints += 1


    def get_checkpoint(self, load_path, prefix, step=None):
        '''Load a checkpoint from a checkpoint file. The checkpoints are
        dictionaries with the model parameters and model state.

        Args:
            load_path: The path to the directory containing the checkpoints
            prefix: Prefix used for the checkpoint file. Used to differentiate
                different tuning runs
            step: Which checkpoint file to load. By default loads the most recent
            model: An instance of the model class. If provided, checkpoint 
                parameters/state will be loaded into the model

        Returns:
            checkpoint (dict): A dictionary containing 'params' and 'model_state'
        '''

        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=load_path,
            target=None,
            step=step,
            prefix=prefix,
        )
        if ckpt is None:
            logging.warn('Could not find file containing prefix %s, \
                in directory %s' % (prefix, load_path))

        return ckpt
