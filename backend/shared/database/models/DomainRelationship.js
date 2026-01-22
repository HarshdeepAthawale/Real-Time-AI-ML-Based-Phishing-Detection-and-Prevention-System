"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DomainRelationship = void 0;
const typeorm_1 = require("typeorm");
const Domain_1 = require("./Domain");
let DomainRelationship = class DomainRelationship {
};
exports.DomainRelationship = DomainRelationship;
__decorate([
    (0, typeorm_1.PrimaryGeneratedColumn)('uuid'),
    __metadata("design:type", String)
], DomainRelationship.prototype, "id", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'uuid' }),
    __metadata("design:type", String)
], DomainRelationship.prototype, "source_domain_id", void 0);
__decorate([
    (0, typeorm_1.ManyToOne)(() => Domain_1.Domain, (domain) => domain.source_relationships, { onDelete: 'CASCADE' }),
    (0, typeorm_1.JoinColumn)({ name: 'source_domain_id' }),
    __metadata("design:type", Domain_1.Domain)
], DomainRelationship.prototype, "source_domain", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'uuid' }),
    __metadata("design:type", String)
], DomainRelationship.prototype, "target_domain_id", void 0);
__decorate([
    (0, typeorm_1.ManyToOne)(() => Domain_1.Domain, (domain) => domain.target_relationships, { onDelete: 'CASCADE' }),
    (0, typeorm_1.JoinColumn)({ name: 'target_domain_id' }),
    __metadata("design:type", Domain_1.Domain)
], DomainRelationship.prototype, "target_domain", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 50 }),
    __metadata("design:type", String)
], DomainRelationship.prototype, "relationship_type", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'decimal', precision: 5, scale: 2, default: 1.0 }),
    __metadata("design:type", Number)
], DomainRelationship.prototype, "strength", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'jsonb', default: {} }),
    __metadata("design:type", Object)
], DomainRelationship.prototype, "metadata", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' }),
    __metadata("design:type", Date)
], DomainRelationship.prototype, "first_seen_at", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' }),
    __metadata("design:type", Date)
], DomainRelationship.prototype, "last_seen_at", void 0);
__decorate([
    (0, typeorm_1.CreateDateColumn)(),
    __metadata("design:type", Date)
], DomainRelationship.prototype, "created_at", void 0);
exports.DomainRelationship = DomainRelationship = __decorate([
    (0, typeorm_1.Entity)('domain_relationships'),
    (0, typeorm_1.Unique)(['source_domain_id', 'target_domain_id', 'relationship_type'])
], DomainRelationship);
