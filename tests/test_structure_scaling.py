"""CPU regression checks for the exact structural experiment."""
import itertools
import unittest

from scripts import structure_scaling as s


class StructureScalingTests(unittest.TestCase):
    def test_signed_convention(self):
        self.assertTrue(s.calibrate(4)["passed"])

    def test_octonion_sums_are_alternative(self):
        for i, j, k in itertools.product(range(8), repeat=3):
            x, y = s.add(s.e(i), s.e(j)), s.e(k)
            self.assertEqual(s.assoc(x, x, y), {})
            self.assertEqual(s.assoc(y, x, x), {})

    def test_sedenion_witness_embeds_at_massive_level(self):
        x, y, z = s.low_witness()
        for n in (4, 34284, 600000):
            r = s.residuals(x, y, z, n)
            self.assertTrue(r["left_alternativity"])
            self.assertFalse(r["anti_involution"])
            self.assertFalse(r["cd_doubling_consistency"])

    def test_high_bit_doubling_and_involutions(self):
        n = 600000
        x = s.add(s.e((1 << (n - 1)) | 5), s.e(3))
        y = s.add(s.e((1 << (n - 2)) | 2), s.e(1 << (n - 1)))
        r = s.residuals(x, y, s.e(7), n)
        for name in ("anti_involution", "scalar_norm", "cd_doubling_consistency",
                     "dimagma_direct_product_anti_involution"):
            self.assertEqual(r[name], {})

    def test_selected_dimagma_is_direct_product(self):
        # At level 1, (0,1)^2=(0,1) in R x R, versus -1 in C.
        self.assertEqual(s.dimagma_product(s.e(1), s.e(1), 1), s.e(1))
        self.assertEqual(s.mul(s.e(1), s.e(1)), {0: -1})


if __name__ == "__main__":
    unittest.main()
