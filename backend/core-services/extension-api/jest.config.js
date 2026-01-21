module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src', '<rootDir>/tests'],
  testMatch: ['**/__tests__/**/*.ts', '**/?(*.)+(spec|test).ts'],
  transform: {
    '^.+\\.ts$': ['ts-jest', {
      tsconfig: {
        target: 'ES2020',
        module: 'commonjs',
        lib: ['ES2020'],
        strict: true,
        esModuleInterop: true,
        skipLibCheck: true,
        forceConsistentCasingInFileNames: true,
        resolveJsonModule: true,
        moduleResolution: 'node',
        experimentalDecorators: true,
        emitDecoratorMetadata: true,
        strictPropertyInitialization: false,
        // Allow importing non-existent modules (Jest will resolve them at runtime)
        noResolve: false,
      },
      isolatedModules: false,
    }],
  },
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/index.ts',
    '!src/**/*.interface.ts',
    '!src/config/**',
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@shared/(.*)$': '<rootDir>/../../shared/$1',
    // Map relative paths like ../../shared/database to actual path (from src/)
    '^(\\.\\./){2}shared/(.*)$': '<rootDir>/../../shared/$2',
    // Map relative paths like ../../../shared/database to actual path (from tests/)
    '^(\\.\\./){3}shared/(.*)$': '<rootDir>/../../shared/$2',
    // Map relative paths like ../../../../shared/database to actual path (from tests/fixtures or tests/helpers)
    '^(\\.\\./){4}shared/(.*)$': '<rootDir>/../../shared/$2',
  },
  modulePathIgnorePatterns: [],
  testTimeout: 10000,
  verbose: true,
  setupFilesAfterEnv: ['<rootDir>/tests/setup.ts'],
  maxWorkers: 1, // Run tests sequentially to avoid deadlocks
};
