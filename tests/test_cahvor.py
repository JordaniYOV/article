"""Geometric tests are synthetic fixtures, not scientific pilot results."""
import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy optional dependency')
class CahvorTests(unittest.TestCase):
    def model(self, center=0):
        return dict(C=[center,0,0],A=[0,0,1],H=[100,0,50],V=[0,100,40],O=[0,0,1],R=[0.01,0.02,0.003],
                    frame='fixture',frame_indices={'SITE':'0'},frame_solution='fixture',
                    axes={'Sample':100,'Line':80})

    def test_distorted_projection_inverse_round_trip(self):
        import numpy as np
        from planetary_vlm.datasets.cahvor import pixel_rays,project_direction
        model=self.model()
        xy=np.array([[50,40],[10,10],[90,70]])
        predicted,valid=project_direction(model,pixel_rays(model,xy))
        np.testing.assert_allclose(predicted,xy,atol=1e-8)
        self.assertTrue(valid.all())

    def test_frame_mismatch_rejected(self):
        from planetary_vlm.datasets.cahvor import stereo_frame
        left,right=self.model(),self.model(0.2)
        right['frame_indices']={'SITE':'1'}
        with self.assertRaises(ValueError):
            stereo_frame(left,right)

    def test_triangulation_principal_point_offset_and_epipolar_lines(self):
        import numpy as np
        from planetary_vlm.datasets.cahvor import disparity_xyz
        params=dict(focal_px=100,baseline_m=0.2,cx_left=40,cx_right=50,cy=40,basis=np.eye(3))
        # A 2m point: physical disparity 10px, rendered disparity 0px.
        xyz,z=disparity_xyz(np.zeros((80,100)),params)
        np.testing.assert_allclose(z,2)
        np.testing.assert_allclose(xyz[40,40],[0,0,2])

    def test_pitch_fit_and_rotation_keep_center(self):
        import numpy as np
        from planetary_vlm.datasets.cahvor import rotate_camera,fit_epipolar_pitch,project_direction
        left,right=self.model(),self.model(0.2)
        tilted=rotate_camera(right,[1,0,0],np.deg2rad(0.03))
        points=np.array([[0,0,2],[0.1,0.2,3],[-0.1,-0.2,4]])
        pl,_=project_direction(left,points)
        # Observations come from the true model; inference starts from a tilted calibration.
        pr,_=project_direction(right,points-np.array(right['C']))
        correction=fit_epipolar_pitch(left,tilted,pl,pr,np.array([1,0,0]))
        self.assertAlmostEqual(np.degrees(correction),-0.03,places=6)
        fixed=rotate_camera(tilted,[1,0,0],correction)
        np.testing.assert_allclose(fixed['A'],right['A'],atol=1e-10)
        self.assertEqual(fixed['C'],right['C'])
