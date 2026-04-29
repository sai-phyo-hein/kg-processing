# RECAP Community Intelligence Platform — Schema Reference

---

## NODE TYPES

### Tambon
| Field | Type | Required |
|---|---|---|
| `tambon_id` | string | yes |
| `tambon_name_th` | string | yes |
| `tambon_name_en` | string | no |
| `district_name` | string | yes |
| `province_name` | string | yes |
| `region` | string | no |
| `population_total` | integer | no |
| `households_total` | integer | no |
| `report_year` | integer | yes |

---

### Village
| Field | Type | Required |
|---|---|---|
| `village_id` | string | yes |
| `tambon_id` | string | yes |
| `village_name` | string | yes |
| `village_no` | integer | no |
| `population` | integer | no |
| `households` | integer | no |
| `notes` | string | no |

---

### SocialCapital
| Field | Type | Required |
|---|---|---|
| `capital_id` | string | yes |
| `capital_name` | string | yes |
| `capital_name_canonical` | string | no |
| `capital_type` | string | yes |
| `capital_level` | enum → `SocialCapitalLevel` | yes |
| `description` | string | no |
| `active_status` | boolean | no |
| `start_year` | integer | no |
| `end_year` | integer | no |
| `tambon_id` | string | yes |
| `village_id` | string | no |
| `evidence_ref` | string | yes |

---

### Domain
| Field | Type | Required |
|---|---|---|
| `domain_id` | string | yes |
| `domain_code` | enum → `DomainCode` | yes |
| `domain_name_th` | string | yes |
| `domain_name_en` | string | yes |

---

### Activity
| Field | Type | Required |
|---|---|---|
| `activity_id` | string | yes |
| `activity_name` | string | yes |
| `domain_id` | string | yes |
| `activity_type` | string | yes |
| `description` | string | no |
| `is_routine` | boolean | yes |
| `is_innovation` | boolean | yes |
| `scope_level` | enum → `ScopeLevel` | yes |
| `tambon_id` | string | yes |
| `village_id` | string | no |
| `start_period` | string | no |
| `end_period` | string | no |
| `status` | string | yes |
| `evidence_ref` | string | yes |

---

### TargetGroup
| Field | Type | Required |
|---|---|---|
| `target_group_id` | string | yes |
| `target_group_name` | string | yes |
| `target_group_category` | string | yes |
| `is_one_of_13_target_groups` | boolean | yes |
| `description` | string | no |

---

### Impact
| Field | Type | Required |
|---|---|---|
| `impact_id` | string | yes |
| `impact_name` | string | yes |
| `impact_type` | enum → `ImpactType` | yes |
| `impact_domain_id` | string | no |
| `description` | string | no |
| `evidence_strength` | enum → `EvidenceStrength` | yes |
| `tambon_id` | string | yes |
| `village_id` | string | no |
| `evidence_ref` | string | yes |

---

### Evidence
| Field | Type | Required |
|---|---|---|
| `evidence_id` | string | yes |
| `source_document_id` | string | yes |
| `page_no` | integer | no |
| `section_name` | string | no |
| `quote_or_summary` | string | yes |
| `collection_method` | string | yes |
| `confidence` | float (0.0–1.0) | yes |
| `coder` | string | yes |
| `date_recorded` | string (ISO) | no |

---

## PHASE 2 NODE TYPES

### Resource
| Field | Type | Required |
|---|---|---|
| `resource_id` | string | yes |
| `resource_name` | string | yes |
| `resource_type` | enum → `ResourceType` | yes |
| `description` | string | no |
| `internal_external` | string | yes |
| `source_org_id` | string | no |

### EnablingFactor
| Field | Type | Required |
|---|---|---|
| `factor_id` | string | yes |
| `factor_name` | string | yes |
| `factor_category` | string | yes |
| `description` | string | no |

### CommunityIssue
| Field | Type | Required |
|---|---|---|
| `issue_id` | string | yes |
| `issue_name` | string | yes |
| `domain_id` | string | yes |
| `issue_type` | string | yes |
| `severity_level` | string | yes |
| `description` | string | no |
| `tambon_id` | string | yes |
| `village_id` | string | no |
| `affected_groups` | string[] | no |
| `evidence_ref` | string | yes |

### CapabilityDimension
| Field | Type | Required |
|---|---|---|
| `capability_id` | string | yes |
| `capability_name` | string | yes |
| `capability_group` | string | yes |
| `description` | string | no |

### CapabilityAssessment
| Field | Type | Required |
|---|---|---|
| `assessment_id` | string | yes |
| `tambon_id` | string | yes |
| `village_id` | string | no |
| `capital_id` | string | no |
| `domain_id` | string | no |
| `capability_id` | string | yes |
| `score` | float | yes |
| `assessment_method` | string | yes |
| `evidence_ref` | string | yes |
| `remarks` | string | no |

### Innovation
| Field | Type | Required |
|---|---|---|
| `innovation_id` | string | yes |
| `innovation_name` | string | yes |
| `description` | string | no |
| `origin_story` | string | no |
| `innovation_level` | string | yes |
| `target_problem` | string | no |
| `tambon_id` | string | yes |
| `village_id` | string | no |
| `status` | string | yes |
| `evidence_ref` | string | yes |

### Actor
| Field | Type | Required |
|---|---|---|
| `actor_id` | string | yes |
| `actor_name` | string | yes |
| `actor_role` | string | yes |
| `actor_category` | string | yes |
| `affiliated_capital_id` | string | no |
| `tambon_id` | string | yes |
| `village_id` | string | no |
| `notes` | string | no |

---

## ENUMERATIONS

### `SocialCapitalLevel`
```
PERSON_FAMILY
SOCIAL_GROUP_COMMUNITY_ORG
AGENCY_RESOURCE_SOURCE
VILLAGE_COMMUNITY
TAMBON
NETWORK
```

### `DomainCode`
```
SOCIAL
ECONOMIC
ENVIRONMENT
HEALTH
GOVERNANCE
```

### `ScopeLevel`
```
VILLAGE
TAMBON
NETWORK
```

### `ImpactType`
```
DIRECT
INDIRECT
SHORT_TERM
LONG_TERM
```

### `EvidenceStrength`
```
STRONG
MODERATE
WEAK
CLAIMED
```

### `ResourceType`
```
PEOPLE
DATA
BUDGET
METHOD
TECHNOLOGY
FACILITY
KNOWLEDGE
NETWORK
POLICY
```

---

## RELATIONS

### MVP
| Subject | Relation | Object |
|---|---|---|
| Village | `BELONGS_TO` | Tambon |
| SocialCapital | `BELONGS_TO` | Tambon |
| SocialCapital | `LOCATED_IN` | Village |
| SocialCapital | `PERFORMS` | Activity |
| Activity | `BELONGS_TO` | Domain |
| Activity | `TARGETS` | TargetGroup |
| Activity | `PRODUCES` | Impact |
| Impact | `AFFECTS` | TargetGroup |
| Evidence | `SUPPORTS` | SocialCapital |
| Evidence | `SUPPORTS` | Activity |
| Evidence | `SUPPORTS` | Impact |

### Phase 2
| Subject | Relation | Object |
|---|---|---|
| Actor | `BELONGS_TO` | SocialCapital |
| Actor | `PARTICIPATES_IN` | Activity |
| Activity | `USES` | Resource |
| Activity | `ENABLED_BY` | EnablingFactor |
| Activity | `ADDRESSES` | CommunityIssue |
| Impact | `STRENGTHENS` | CapabilityDimension |
| SocialCapital | `CONNECTED_TO` | SocialCapital |
| Innovation | `EMERGES_FROM` | Activity |

---

## TRIPLE OUTPUT FORMAT

```json
{
  "extraction_id": "ext-001",
  "source_document_id": "doc-001",
  "tambon_id": "tambon_kokkhan",
  "primary_node_type": "SocialCapital",
  "primary_node": {
    "capital_id": "sc-001",
    "capital_name": "กลุ่มอาสาสมัครดูแลผู้สูงอายุ",
    "capital_name_canonical": "กลุ่มอาสาสมัคร",
    "capital_type": "กลุ่ม",
    "capital_level": "SOCIAL_GROUP_COMMUNITY_ORG",
    "active_status": true,
    "tambon_id": "tambon_kokkhan",
    "village_id": "village_moo1_kokkhan",
    "evidence_ref": "ev-001"
  },
  "relations": [
    { "label": "BELONGS_TO",  "object_type": "Tambon",   "object_id": "tambon_kokkhan" },
    { "label": "LOCATED_IN",  "object_type": "Village",  "object_id": "village_moo1_kokkhan" },
    { "label": "PERFORMS",    "object_type": "Activity", "object_id": "act-001" }
  ],
  "evidence": {
    "evidence_id": "ev-001",
    "source_document_id": "doc-001",
    "page_no": 12,
    "section_name": "ด้านสังคม",
    "quote_or_summary": "กลุ่มอาสาสมัครช่วยดูแลผู้สูงอายุในหมู่ที่ 1 โดยมีสมาชิก 15 คน",
    "collection_method": "recap_summary",
    "confidence": 0.85,
    "coder": "agent-v1",
    "date_recorded": "2024-01-15"
  }
}
```

---

## INGESTION PAYLOAD FORMAT

```json
{
  "tambon": "เกาะขันธ์",
  "village": "หมู่ 1",
  "domain": "SOCIAL",
  "social_capital": {
    "name": "กลุ่มอาสาสมัคร",
    "level": "SOCIAL_GROUP_COMMUNITY_ORG",
    "description": "กลุ่มช่วยเหลือดูแลในชุมชน"
  },
  "activity": {
    "name": "การดูแลช่วยเหลือผู้สูงอายุ",
    "type": "care_support",
    "scope_level": "VILLAGE",
    "is_routine": true,
    "is_innovation": false,
    "status": "active"
  },
  "target_groups": ["ผู้สูงอายุ"],
  "impacts": [
    { "name": "เข้าถึงการช่วยเหลือ", "type": "DIRECT",     "evidence_strength": "MODERATE" },
    { "name": "สุขภาพดีขึ้น",        "type": "SHORT_TERM", "evidence_strength": "CLAIMED"  }
  ],
  "evidence": {
    "source_document_id": "doc-001",
    "page_no": 12,
    "section_name": "ด้านสังคม",
    "quote_or_summary": "กลุ่มอาสาสมัครช่วยดูแลผู้สูงอายุในหมู่ที่ 1",
    "collection_method": "recap_summary",
    "confidence": 0.85,
    "coder": "agent-v1"
  }
}
```
