"""Batch-one device transfer with optional host rasterizer scalar arguments."""


def target_to_device(common, target, device, host_camera_scalars=False):
    target = common.collate1(target)
    saved = {}
    if host_camera_scalars:
        # Preserve collate1's dtype rounding. Rasterizer reads these four
        # values with int()/float(); copying them to CUDA first forces syncs.
        camera = target['render_cam_params']
        for key in ('image_height', 'image_width', 'tanfovx', 'tanfovy'):
            saved[key] = camera.pop(key)
    target = common.to_dev(target, device)
    target['render_cam_params'].update(saved)
    return target
