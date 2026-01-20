import { z } from 'zod';

export const detectEmailSchema = z.object({
  emailContent: z.string().min(1, 'Email content is required'),
  organizationId: z.string().uuid().optional(),
  includeFeatures: z.boolean().optional()
});

export const detectURLSchema = z.object({
  url: z.string().url('Invalid URL format'),
  legitimateDomain: z.string().optional(),
  legitimateUrl: z.string().url('Invalid legitimate URL format').optional(),
  organizationId: z.string().uuid().optional()
});

export const detectTextSchema = z.object({
  text: z.string().min(1, 'Text is required'),
  includeFeatures: z.boolean().optional(),
  organizationId: z.string().uuid().optional()
});
