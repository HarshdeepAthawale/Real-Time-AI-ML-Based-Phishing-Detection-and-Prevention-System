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
exports.ThreatIntelligenceFeed = void 0;
const typeorm_1 = require("typeorm");
const IOC_1 = require("./IOC");
let ThreatIntelligenceFeed = class ThreatIntelligenceFeed {
};
exports.ThreatIntelligenceFeed = ThreatIntelligenceFeed;
__decorate([
    (0, typeorm_1.PrimaryGeneratedColumn)('uuid'),
    __metadata("design:type", String)
], ThreatIntelligenceFeed.prototype, "id", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 255, unique: true }),
    __metadata("design:type", String)
], ThreatIntelligenceFeed.prototype, "name", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 50 }),
    __metadata("design:type", String)
], ThreatIntelligenceFeed.prototype, "feed_type", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'text', nullable: true }),
    __metadata("design:type", String)
], ThreatIntelligenceFeed.prototype, "api_endpoint", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'text', nullable: true }),
    __metadata("design:type", String)
], ThreatIntelligenceFeed.prototype, "api_key_encrypted", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'int', default: 60 }),
    __metadata("design:type", Number)
], ThreatIntelligenceFeed.prototype, "sync_interval_minutes", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'timestamp', nullable: true }),
    __metadata("design:type", Date)
], ThreatIntelligenceFeed.prototype, "last_sync_at", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 20, nullable: true }),
    __metadata("design:type", String)
], ThreatIntelligenceFeed.prototype, "last_sync_status", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'text', nullable: true }),
    __metadata("design:type", String)
], ThreatIntelligenceFeed.prototype, "last_sync_error", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'boolean', default: true }),
    __metadata("design:type", Boolean)
], ThreatIntelligenceFeed.prototype, "is_active", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'decimal', precision: 5, scale: 2, default: 50 }),
    __metadata("design:type", Number)
], ThreatIntelligenceFeed.prototype, "reliability_score", void 0);
__decorate([
    (0, typeorm_1.CreateDateColumn)(),
    __metadata("design:type", Date)
], ThreatIntelligenceFeed.prototype, "created_at", void 0);
__decorate([
    (0, typeorm_1.UpdateDateColumn)(),
    __metadata("design:type", Date)
], ThreatIntelligenceFeed.prototype, "updated_at", void 0);
__decorate([
    (0, typeorm_1.OneToMany)(() => IOC_1.IOC, (ioc) => ioc.feed),
    __metadata("design:type", Array)
], ThreatIntelligenceFeed.prototype, "iocs", void 0);
exports.ThreatIntelligenceFeed = ThreatIntelligenceFeed = __decorate([
    (0, typeorm_1.Entity)('threat_intelligence_feeds')
], ThreatIntelligenceFeed);
