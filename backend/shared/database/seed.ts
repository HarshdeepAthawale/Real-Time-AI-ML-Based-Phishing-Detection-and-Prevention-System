import { DataSource } from 'typeorm';
import bcrypt from 'bcrypt';
import { connectPostgreSQL } from './connection';
import { Organization } from './models/Organization';
import { User } from './models/User';

export const seedDatabase = async (): Promise<void> => {
  const dataSource = await connectPostgreSQL();
  const orgRepository = dataSource.getRepository(Organization);
  const userRepository = dataSource.getRepository(User);

  // Check if default organization exists
  let defaultOrg = await orgRepository.findOne({
    where: { domain: 'example.com' },
  });

  if (!defaultOrg) {
    // Create default organization
    defaultOrg = orgRepository.create({
      name: 'Example Organization',
      domain: 'example.com',
      plan: 'enterprise',
      max_users: 100,
      max_api_calls_per_day: 100000,
    });
    await orgRepository.save(defaultOrg);
    console.log('Created default organization');
  }

  // Check if admin user exists
  const adminUser = await userRepository.findOne({
    where: { email: 'admin@example.com' },
  });

  if (!adminUser) {
    // Create admin user
    const passwordHash = await bcrypt.hash('admin123', 10);
    const admin = userRepository.create({
      organization_id: defaultOrg.id,
      email: 'admin@example.com',
      password_hash: passwordHash,
      first_name: 'Admin',
      last_name: 'User',
      role: 'admin',
      is_active: true,
    });
    await userRepository.save(admin);
    console.log('Created admin user (email: admin@example.com, password: admin123)');
  }

  console.log('Database seeding completed');
};

// Run if called directly
if (require.main === module) {
  seedDatabase()
    .then(() => {
      console.log('Seeding finished');
      process.exit(0);
    })
    .catch((error) => {
      console.error('Seeding failed:', error);
      process.exit(1);
    });
}
