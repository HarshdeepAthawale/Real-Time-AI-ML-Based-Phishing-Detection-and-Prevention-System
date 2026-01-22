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
exports.ThreatIndicator = void 0;
const typeorm_1 = require("typeorm");
const Threat_1 = require("./Threat");
let ThreatIndicator = class ThreatIndicator {
};
exports.ThreatIndicator = ThreatIndicator;
__decorate([
    (0, typeorm_1.PrimaryGeneratedColumn)('uuid'),
    __metadata("design:type", String)
], ThreatIndicator.prototype, "id", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'uuid' }),
    __metadata("design:type", String)
], ThreatIndicator.prototype, "threat_id", void 0);
__decorate([
    (0, typeorm_1.ManyToOne)(() => Threat_1.Threat, (threat) => threat.indicators, { onDelete: 'CASCADE' }),
    (0, typeorm_1.JoinColumn)({ name: 'threat_id' }),
    __metadata("design:type", Threat_1.Threat)
], ThreatIndicator.prototype, "threat", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 50 }),
    __metadata("design:type", String)
], ThreatIndicator.prototype, "indicator_type", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'text' }),
    __metadata("design:type", String)
], ThreatIndicator.prototype, "indicator_value", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'varchar', length: 50, nullable: true }),
    __metadata("design:type", String)
], ThreatIndicator.prototype, "source", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' }),
    __metadata("design:type", Date)
], ThreatIndicator.prototype, "first_seen_at", void 0);
__decorate([
    (0, typeorm_1.Column)({ type: 'timestamp', default: () => 'CURRENT_TIMESTAMP' }),
    __metadata("design:type", Date)
], ThreatIndicator.prototype, "last_seen_at", void 0);
__decorate([
    (0, typeorm_1.CreateDateColumn)(),
    __metadata("design:type", Date)
], ThreatIndicator.prototype, "created_at", void 0);
exports.ThreatIndicator = ThreatIndicator = __decorate([
    (0, typeorm_1.Entity)('threat_indicators')
], ThreatIndicator);
