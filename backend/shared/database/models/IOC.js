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
exports.IOC = void 0;
const typeorm_1 = require("typeorm");
const ThreatIntelligenceFeed_1 = require("./ThreatIntelligenceFeed");
const IOCMatch_1 = require("./IOCMatch");
let IOC = class IOC {
};
exports.IOC = IOC;
__decorate([
    (0, typeorm_1.PrimaryGeneratedColumn)('uuid'),
    __metadata("design:type", String)
], IOC.prototype, "id", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'uuid', nullable: true }),
    __metadata("design:type", String)
], IOC.prototype, "feed_id", void 0);
__decorate([
    (0, typeorm_1.ManyToOne)(() => ThreatIntelligenceFeed_1.ThreatIntelligenceFeed, (feed) => feed.iocs, { onDelete: 'SET NULL', nullable: true }),
    (0, typeorm_1.JoinColumn)({ name: 'feed_id' }),
    __metadata("design:type", ThreatIntelligenceFeed_1.ThreatIntelligenceFeed)
], IOC.prototype, "feed", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 50 }),
    __metadata("design:type", String)
], IOC.prototype, "ioc_type", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'text' }),
    __metadata("design:type", String)
], IOC.prototype, "ioc_value", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 64, nullable: true }),
    __metadata("design:type", String)
], IOC.prototype, "ioc_value_hash", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 100, nullable: true }),
    __metadata("design:type", String)
], IOC.prototype, "threat_type", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 20, nullable: true }),
    __metadata("design:type", String)
], IOC.prototype, "severity", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'decimal', precision: 5, scale: 2, nullable: true }),
    __metadata("design:type", Number)
], IOC.prototype, "confidence", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'timestamp', nullable: true }),
    __metadata("design:type", Date)
], IOC.prototype, "first_seen_at", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'timestamp', nullable: true }),
    __metadata("design:type", Date)
], IOC.prototype, "last_seen_at", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'int', default: 1 }),
    __metadata("design:type", Number)
], IOC.prototype, "source_reports", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'jsonb', default: {} }),
    __metadata("design:type", Object)
], IOC.prototype, "metadata", void 0);
__decorate([
    (0, typeorm_1.CreateDateColumn)(),
    __metadata("design:type", Date)
], IOC.prototype, "created_at", void 0);
__decorate([
    (0, typeorm_1.UpdateDateColumn)(),
    __metadata("design:type", Date)
], IOC.prototype, "updated_at", void 0);
__decorate([
    (0, typeorm_1.OneToMany)(() => IOCMatch_1.IOCMatch, (match) => match.ioc),
    __metadata("design:type", Array)
], IOC.prototype, "matches", void 0);
exports.IOC = IOC = __decorate([
    (0, typeorm_1.Entity)('iocs'),
    (0, typeorm_1.Unique)(['ioc_type', 'ioc_value_hash'])
], IOC);
