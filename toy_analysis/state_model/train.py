from toy_diffusion_models.state_diffusion_model import main

if __name__ == '__main__':
    import jaynes
    from ml_logger import logger, instr
    from dd_launch import RUN

    jaynes.config('tjlab-gpu')

    RUN.CUDA_VISIBLE_DEVICES = "3"

    thunk = instr(main)
    logger.log_text("""
    charts:
    - yKey: loss/mean
      xKey: epoch
    - type: image
      glob: samples/ep_20/frame_*.png
    - type: image
      glob: samples/ep_40/frame_*.png
    """, ".charts.yml", dedent=True, overwrite=True)
    jaynes.run(thunk)
    jaynes.listen()
