import { describe, expect, it } from 'vitest';
import { cleanProtocol, csvCell, metricValue } from '../domain';
import { condition, run } from './fixtures';

describe('scientific display contracts', () => {
  it('keeps absent quality as null while preserving measured zeros', () => {
    expect(metricValue(undefined, 'accuracy')).toBeNull();
    expect(metricValue({ ...condition, quality: null }, 'mean_iou')).toBeNull();
    expect(metricValue({ ...condition, quality: { accuracy: 0, coverage: 0 } }, 'accuracy')).toBe(0);
    expect(metricValue({ ...condition, quality: { coverage: 0 } }, 'coverage')).toBe(0);
  });
  it('uses native segmentation coverage and foreground Dice without confusing QA', () => {
    const value = { ...condition, quality: { prediction_coverage: .6, per_class: { IMP: { dice: .73 } } } };
    expect(metricValue(value, 'coverage')).toBe(.6);
    expect(metricValue(value, 'dice')).toBe(.73);
    expect(metricValue(value, 'accuracy')).toBeNull();
  });
  it('copies the ordered reusable protocol, removing internal metadata', () => {
    const copied = cleanProtocol({ ...run.protocol, ...{ selection_sha256: 'internal', model_config_sha256: 'internal' } });
    expect(copied.distortions.at(-1)).toEqual(['shear', 'hard_shadows', 'lens_flare', 'radiation']);
    expect(copied).not.toHaveProperty('selection_sha256');
    expect(copied.distortions[0]).not.toBe(run.protocol.distortions[0]);
  });
  it('exports safe CSV cells with embedded quotes and formula-like names', () => {
    expect(csvCell('a"b')).toBe('"a""b"');
    expect(csvCell('=cmd')).toBe('"\'=cmd"');
    expect(csvCell(null)).toBe('""');
  });
});
