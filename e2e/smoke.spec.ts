import { test, expect } from '@playwright/test';

test.describe('Smoke tests - core pages load', () => {
  test('Dashboard loads', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByRole('heading', { name: /security dashboard/i })).toBeVisible({ timeout: 10000 });
  });

  test('Detection page loads', async ({ page }) => {
    await page.goto('/detection');
    await expect(page.getByRole('heading', { name: /threat detection/i })).toBeVisible({ timeout: 10000 });
  });

  test('Intelligence page loads', async ({ page }) => {
    await page.goto('/intelligence');
    await expect(page.getByRole('heading', { name: /threat intelligence/i })).toBeVisible({ timeout: 10000 });
  });

  test('Settings page loads', async ({ page }) => {
    await page.goto('/settings');
    await expect(page.getByRole('heading', { name: /settings/i })).toBeVisible({ timeout: 10000 });
  });

  test('IOCs page loads', async ({ page }) => {
    await page.goto('/iocs');
    await expect(page.getByRole('heading', { name: /ioc management/i })).toBeVisible({ timeout: 10000 });
  });

  test('Monitoring page loads', async ({ page }) => {
    await page.goto('/monitoring');
    await expect(page.getByRole('heading', { name: /real-time monitoring/i })).toBeVisible({ timeout: 10000 });
  });

  test('Sandbox page loads', async ({ page }) => {
    await page.goto('/sandbox');
    await expect(page.getByRole('heading', { name: /sandbox analysis/i })).toBeVisible({ timeout: 10000 });
  });

  test('Feeds page loads', async ({ page }) => {
    await page.goto('/feeds');
    await expect(page.getByRole('heading', { name: /threat feeds/i })).toBeVisible({ timeout: 10000 });
  });
});
