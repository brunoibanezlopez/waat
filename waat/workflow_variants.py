"""Generate domain-realistic workflow variants for scaling experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


TARGET_NODE_COUNTS = (6, 20, 50, 100)


def workflow_node_count(workflow_data: dict[str, Any]) -> int:
    """Count processing states plus terminal states."""
    return len(workflow_data["states"]) + len(set(workflow_data["terminal_states"]))


def build_variant(base_workflow: dict[str, Any], target_nodes: int) -> dict[str, Any]:
    """Return a structurally valid telecom support workflow with target_nodes nodes."""
    if target_nodes == 6:
        workflow = deepcopy(base_workflow)
        workflow["workflow"] = "SERVICE_REQUEST_6_NODES"
        return workflow
    if target_nodes not in TARGET_NODE_COUNTS:
        raise ValueError(f"Unsupported workflow size: {target_nodes}")

    workflow = {
        "workflow": f"SERVICE_REQUEST_{target_nodes}_NODES",
        "initial_state": "CLASSIFY_REQUEST",
        "terminal_states": ["REQUEST_COMPLETE", "REQUEST_ESCALATED"],
        "states": {},
    }
    _add_core_states(workflow)
    if target_nodes >= 50:
        _add_50_node_states(workflow)
    if target_nodes >= 100:
        _add_100_node_states(workflow)

    actual_nodes = workflow_node_count(workflow)
    if actual_nodes != target_nodes:
        raise AssertionError(f"Expected {target_nodes} nodes, generated {actual_nodes}")
    return workflow


def write_variants(base_path: Path, output_dir: Path, target_counts: tuple[int, ...] = TARGET_NODE_COUNTS) -> list[Path]:
    """Write YAML workflow variants and return their paths."""
    base_workflow = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for target_count in target_counts:
        variant = build_variant(base_workflow, target_count)
        path = output_dir / f"service_request_{target_count}_nodes.yaml"
        path.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def _add_core_states(workflow: dict[str, Any]) -> None:
    """Add an 18-state, 20-node telecom service workflow."""
    add = _state_adder(workflow)
    add(
        "CLASSIFY_REQUEST",
        "call_intake_classifier_agent",
        [
            ("VERIFY_CUSTOMER_IDENTITY", "request maps to account, billing, service, complaint, or field-support handling"),
            ("REVIEW_SECURITY_OR_PRIVACY_RISK", "request contains sensitive data, account takeover risk, or policy-bypass language"),
            ("REQUEST_ESCALATED", "request cannot be classified with sufficient confidence"),
        ],
    )
    add(
        "VERIFY_CUSTOMER_IDENTITY",
        "call_identity_verification_agent",
        [
            ("RETRIEVE_CUSTOMER_PROFILE", "customer identity is verified or low-risk assisted service is allowed"),
            ("REVIEW_SECURITY_OR_PRIVACY_RISK", "identity verification fails or request involves another customer's data"),
        ],
    )
    add(
        "RETRIEVE_CUSTOMER_PROFILE",
        "call_customer_profile_agent",
        [
            ("CHECK_ACCOUNT_STATUS", "request is about account information, payment status, ownership, or billing"),
            ("CHECK_CONTRACT_TERMS", "request is a service modification, upgrade, relocation, or contract exception"),
            ("CAPTURE_COMPLAINT_DETAILS", "request is primarily a complaint or dissatisfaction event"),
            ("REQUEST_ESCALATED", "profile is unavailable or does not match the requesting party"),
        ],
    )
    add(
        "REVIEW_SECURITY_OR_PRIVACY_RISK",
        "call_compliance_agent",
        [
            ("REQUEST_ESCALATED", "privacy, security, or policy risk requires manual handling"),
            ("VERIFY_CUSTOMER_IDENTITY", "risk is cleared and the request may continue through normal identity checks"),
        ],
    )
    add(
        "CHECK_ACCOUNT_STATUS",
        "call_account_status_agent",
        [
            ("HANDLE_ACCOUNT_QUERY", "account is active and the requested data is available"),
            ("ASSESS_BILLING_DISPUTE", "account request contains payment or balance-dispute indicators"),
            ("CREATE_SUPPORT_TICKET", "account data is unavailable, inconsistent, or requires back-office recovery"),
        ],
    )
    add(
        "HANDLE_ACCOUNT_QUERY",
        "call_account_agent",
        [
            ("SEND_CUSTOMER_CONFIRMATION", "query answered and customer is satisfied"),
            ("ASSESS_BILLING_DISPUTE", "answer reveals a payment, charge, or statement dispute"),
            ("CREATE_SUPPORT_TICKET", "account data cannot be retrieved automatically"),
        ],
    )
    add(
        "ASSESS_BILLING_DISPUTE",
        "call_billing_dispute_agent",
        [
            ("HANDLE_COMPLAINT", "billing dispute can be addressed through complaint-resolution handling"),
            ("REQUEST_MANUAL_APPROVAL", "refund, credit, or adjustment exceeds automated authority"),
            ("CREATE_SUPPORT_TICKET", "billing records are incomplete or require back-office investigation"),
        ],
    )
    add(
        "CHECK_CONTRACT_TERMS",
        "call_contract_agent",
        [
            ("HANDLE_SERVICE_CHANGE", "requested change is permitted under the active contract"),
            ("SCHEDULE_TECHNICIAN", "relocation or installation request requires field scheduling"),
            ("REQUEST_MANUAL_APPROVAL", "change exceeds contract limits or requires commercial approval"),
            ("CREATE_SUPPORT_TICKET", "contract data is missing or inconsistent"),
        ],
    )
    add(
        "HANDLE_SERVICE_CHANGE",
        "call_service_agent",
        [
            ("SEND_CUSTOMER_CONFIRMATION", "service change completed successfully"),
            ("REQUEST_MANUAL_APPROVAL", "change requires manual approval or custom design"),
            ("CREATE_SUPPORT_TICKET", "service platform cannot complete the change automatically"),
        ],
    )
    add(
        "REQUEST_MANUAL_APPROVAL",
        "call_approval_agent",
        [
            ("REQUEST_ESCALATED", "manual approval is required before the customer request can proceed"),
            ("SEND_CUSTOMER_CONFIRMATION", "approval is granted and customer-facing confirmation can be sent"),
        ],
    )
    add(
        "CAPTURE_COMPLAINT_DETAILS",
        "call_complaint_intake_agent",
        [
            ("CHECK_SERVICE_OUTAGE", "complaint concerns service quality, outage, speed, installation, or technician attendance"),
            ("HANDLE_COMPLAINT", "complaint can proceed directly to resolution handling"),
            ("REQUEST_MANUAL_APPROVAL", "complaint indicates regulatory, executive, or unresolved escalation risk"),
        ],
    )
    add(
        "HANDLE_COMPLAINT",
        "call_complaint_agent",
        [
            ("SEND_CUSTOMER_CONFIRMATION", "resolution offered and accepted"),
            ("REQUEST_MANUAL_APPROVAL", "customer remains dissatisfied or issue requires specialist escalation"),
            ("CREATE_SUPPORT_TICKET", "resolution requires follow-up work by another team"),
        ],
    )
    add(
        "CHECK_SERVICE_OUTAGE",
        "call_outage_agent",
        [
            ("SCHEDULE_TECHNICIAN", "field visit or installation remediation is required"),
            ("HANDLE_COMPLAINT", "service issue can be resolved or compensated without field dispatch"),
            ("CREATE_SUPPORT_TICKET", "network investigation is required before resolution"),
        ],
    )
    add(
        "SCHEDULE_TECHNICIAN",
        "call_field_service_agent",
        [
            ("CREATE_SUPPORT_TICKET", "technician appointment is scheduled and must be tracked"),
            ("REQUEST_MANUAL_APPROVAL", "field dispatch requires approval or specialist coordination"),
        ],
    )
    add(
        "CREATE_SUPPORT_TICKET",
        "call_ticketing_agent",
        [
            ("REQUEST_ESCALATED", "ticket requires manual queue ownership or specialist follow-up"),
            ("SEND_CUSTOMER_CONFIRMATION", "ticket created and customer can be notified of next steps"),
        ],
    )
    add(
        "SEND_CUSTOMER_CONFIRMATION",
        "call_notification_agent",
        [
            ("LOG_INTERACTION", "customer confirmation has been sent successfully"),
            ("CREATE_SUPPORT_TICKET", "confirmation failed and follow-up tracking is required"),
        ],
    )
    add(
        "LOG_INTERACTION",
        "call_audit_log_agent",
        [
            ("CLOSE_REQUEST", "interaction notes, decision rationale, and outcome are recorded"),
            ("CREATE_SUPPORT_TICKET", "audit log cannot be completed automatically"),
        ],
    )
    add(
        "CLOSE_REQUEST",
        "call_closure_agent",
        [
            ("REQUEST_COMPLETE", "all required customer-facing and audit steps are complete"),
            ("REQUEST_ESCALATED", "closure check identifies unresolved operational risk"),
        ],
    )


def _add_50_node_states(workflow: dict[str, Any]) -> None:
    add = _state_adder(workflow)
    add("VALIDATE_ACCOUNT_CONTACTS", "call_contact_validation_agent", [("HANDLE_ACCOUNT_QUERY", "contact details are confirmed"), ("CREATE_SUPPORT_TICKET", "contact records require correction")])
    add("RETRIEVE_BILLING_PROFILE", "call_billing_profile_agent", [("HANDLE_ACCOUNT_QUERY", "billing profile supports a standard account answer"), ("VALIDATE_PAYMENT_HISTORY", "payment history or charges must be checked")])
    add("VALIDATE_PAYMENT_HISTORY", "call_payment_history_agent", [("ASSESS_BILLING_DISPUTE", "payment history confirms a dispute or mismatch"), ("SEND_CUSTOMER_CONFIRMATION", "payment history resolves the question")])
    add("REVIEW_BILLING_ADJUSTMENT", "call_credit_assessment_agent", [("REQUEST_MANUAL_APPROVAL", "credit or refund exceeds automated threshold"), ("HANDLE_COMPLAINT", "adjustment can be offered as complaint resolution")])
    add("VERIFY_ACCOUNT_OWNERSHIP", "call_ownership_agent", [("HANDLE_ACCOUNT_QUERY", "ownership is verified"), ("CREATE_SUPPORT_TICKET", "ownership records require back-office review")])
    add("RECOVER_MISSING_ACCOUNT_DATA", "call_data_recovery_agent", [("CREATE_SUPPORT_TICKET", "data recovery requires manual queue handling"), ("HANDLE_ACCOUNT_QUERY", "missing data is recovered")])
    add("VERIFY_SERVICE_INVENTORY", "call_inventory_agent", [("ASSESS_CHANGE_ELIGIBILITY", "service inventory supports the requested change"), ("CREATE_SUPPORT_TICKET", "inventory is missing or inconsistent")])
    add("ASSESS_CHANGE_ELIGIBILITY", "call_change_eligibility_agent", [("HANDLE_SERVICE_CHANGE", "change is eligible for automated fulfilment"), ("REQUEST_MANUAL_APPROVAL", "change requires commercial or technical approval")])
    add("CHECK_SITE_SERVICEABILITY", "call_serviceability_agent", [("SCHEDULE_TECHNICIAN", "site visit or installation appointment is required"), ("HANDLE_SERVICE_CHANGE", "site is serviceable without field work")])
    add("DESIGN_CUSTOM_NETWORK_OPTION", "call_solution_design_agent", [("REQUEST_MANUAL_APPROVAL", "custom design must be reviewed by engineering"), ("CREATE_SUPPORT_TICKET", "design work item is created for solutioning")])
    add("CONFIGURE_SERVICE_ORDER", "call_order_config_agent", [("SUBMIT_SERVICE_ORDER", "order configuration is valid"), ("CREATE_SUPPORT_TICKET", "order configuration failed validation")])
    add("SUBMIT_SERVICE_ORDER", "call_order_submission_agent", [("SEND_CUSTOMER_CONFIRMATION", "service order is submitted"), ("CREATE_SUPPORT_TICKET", "order submission failed and requires follow-up")])
    add("CHECK_SERVICE_HEALTH", "call_service_health_agent", [("RUN_LINE_TEST", "service-health signals require technical diagnostics"), ("HANDLE_COMPLAINT", "health check supports complaint resolution")])
    add("RUN_LINE_TEST", "call_line_test_agent", [("SCHEDULE_TECHNICIAN", "line test indicates premises or field fault"), ("HANDLE_COMPLAINT", "line test does not require dispatch")])
    add("CHECK_APPOINTMENT_HISTORY", "call_appointment_agent", [("SCHEDULE_TECHNICIAN", "new or replacement appointment is required"), ("HANDLE_COMPLAINT", "appointment issue can be compensated without dispatch")])
    add("ASSESS_SERVICE_CREDIT", "call_service_credit_agent", [("HANDLE_COMPLAINT", "service credit can resolve the complaint"), ("REQUEST_MANUAL_APPROVAL", "credit requires approval")])
    add("REVIEW_REGULATORY_RISK", "call_regulatory_agent", [("REQUEST_MANUAL_APPROVAL", "formal complaint or regulatory language requires escalation"), ("HANDLE_COMPLAINT", "risk is low enough for standard complaint handling")])
    add("CHECK_RETENTION_ELIGIBILITY", "call_retention_agent", [("ASSESS_RETENTION_OFFER", "customer is eligible for save or goodwill offer"), ("REQUEST_MANUAL_APPROVAL", "retention exception requires approval")])
    add("ASSESS_RETENTION_OFFER", "call_offer_agent", [("SEND_CUSTOMER_CONFIRMATION", "offer is accepted and can be confirmed"), ("HANDLE_COMPLAINT", "offer is rejected and complaint handling continues")])
    add("OPEN_CASE_RECORD", "call_case_record_agent", [("CREATE_SUPPORT_TICKET", "case record requires queue ownership"), ("SEND_CUSTOMER_CONFIRMATION", "case record is complete and customer can be notified")])
    add("ASSIGN_SPECIALIST_QUEUE", "call_queue_assignment_agent", [("REQUEST_ESCALATED", "specialist queue accepts ownership"), ("CREATE_SUPPORT_TICKET", "queue assignment requires ticket tracking")])
    add("NOTIFY_BACK_OFFICE", "call_back_office_agent", [("CREATE_SUPPORT_TICKET", "back-office team must act asynchronously"), ("SEND_CUSTOMER_CONFIRMATION", "back-office notification is recorded")])
    add("CHECK_KNOWLEDGE_BASE", "call_knowledge_agent", [("HANDLE_ACCOUNT_QUERY", "knowledge article answers the request"), ("CREATE_SUPPORT_TICKET", "knowledge article does not cover the case")])
    add("VALIDATE_CUSTOMER_CONSENT", "call_consent_agent", [("SEND_CUSTOMER_CONFIRMATION", "customer consent is present for the action"), ("REVIEW_SECURITY_OR_PRIVACY_RISK", "consent is missing or invalid")])
    add("REVIEW_VULNERABLE_CUSTOMER_FLAG", "call_vulnerability_agent", [("REQUEST_MANUAL_APPROVAL", "vulnerability marker requires assisted handling"), ("VERIFY_CUSTOMER_IDENTITY", "normal process may continue")])
    add("CHECK_FRAUD_SIGNALS", "call_fraud_agent", [("REVIEW_SECURITY_OR_PRIVACY_RISK", "fraud indicators are present"), ("VERIFY_CUSTOMER_IDENTITY", "fraud checks are clear")])
    add("PREPARE_CUSTOMER_SUMMARY", "call_summary_agent", [("SEND_CUSTOMER_CONFIRMATION", "summary is ready for customer confirmation"), ("LOG_INTERACTION", "summary is internal-only and must be logged")])
    add("RECONCILE_CASE_NOTES", "call_case_notes_agent", [("LOG_INTERACTION", "case notes are reconciled"), ("CREATE_SUPPORT_TICKET", "case notes require manual correction")])
    add("QUALITY_REVIEW_SAMPLE", "call_quality_review_agent", [("CLOSE_REQUEST", "case passes quality review"), ("REQUEST_MANUAL_APPROVAL", "quality review requires supervisor action")])
    add("SURVEY_CUSTOMER_SATISFACTION", "call_survey_agent", [("CLOSE_REQUEST", "survey invitation is sent"), ("LOG_INTERACTION", "survey cannot be sent but outcome is logged")])

    _replace_transitions(workflow, "VALIDATE_ACCOUNT_CONTACTS", [("SEND_CUSTOMER_CONFIRMATION", "validated contact records allow customer confirmation"), ("CREATE_SUPPORT_TICKET", "contact records require correction")])
    _replace_transitions(workflow, "RETRIEVE_BILLING_PROFILE", [("HANDLE_ACCOUNT_QUERY", "billing profile supports a standard account answer"), ("VALIDATE_PAYMENT_HISTORY", "payment history or charges must be checked")])
    _replace_transitions(workflow, "VALIDATE_PAYMENT_HISTORY", [("ASSESS_BILLING_DISPUTE", "payment history confirms a dispute or mismatch"), ("SEND_CUSTOMER_CONFIRMATION", "payment history resolves the question")])
    _replace_transitions(workflow, "REVIEW_BILLING_ADJUSTMENT", [("HANDLE_COMPLAINT", "approved adjustment can be offered as complaint resolution"), ("REQUEST_MANUAL_APPROVAL", "credit or refund exceeds automated threshold")])
    _replace_transitions(workflow, "CONFIGURE_SERVICE_ORDER", [("SUBMIT_SERVICE_ORDER", "order configuration is valid"), ("CREATE_SUPPORT_TICKET", "order configuration failed validation")])
    _replace_transitions(workflow, "SUBMIT_SERVICE_ORDER", [("SEND_CUSTOMER_CONFIRMATION", "service order is submitted"), ("CREATE_SUPPORT_TICKET", "order submission failed and requires follow-up")])
    _replace_transitions(workflow, "PREPARE_CUSTOMER_SUMMARY", [("LOG_INTERACTION", "summary is prepared and should be logged"), ("SEND_CUSTOMER_CONFIRMATION", "summary requires customer-facing confirmation")])
    _replace_transitions(workflow, "RECONCILE_CASE_NOTES", [("CLOSE_REQUEST", "case notes are reconciled and the request can be closed"), ("CREATE_SUPPORT_TICKET", "case notes require manual correction")])

    _prepend_transition(workflow, "CHECK_ACCOUNT_STATUS", "RETRIEVE_BILLING_PROFILE", "billing profile should be retrieved before answering account or payment questions")
    _prepend_transition(workflow, "RETRIEVE_BILLING_PROFILE", "RECOVER_MISSING_ACCOUNT_DATA", "missing account data requires recovery before a response can be given")
    _prepend_transition(workflow, "HANDLE_ACCOUNT_QUERY", "VALIDATE_ACCOUNT_CONTACTS", "account response requires contact-data validation")
    _prepend_transition(workflow, "ASSESS_BILLING_DISPUTE", "REVIEW_BILLING_ADJUSTMENT", "billing dispute requires adjustment or goodwill review")
    _prepend_transition(workflow, "CHECK_CONTRACT_TERMS", "VERIFY_SERVICE_INVENTORY", "service inventory must be verified before change eligibility")
    _prepend_transition(workflow, "CHECK_CONTRACT_TERMS", "CHECK_SITE_SERVICEABILITY", "relocation requests require site serviceability checks")
    _prepend_transition(workflow, "CHECK_CONTRACT_TERMS", "DESIGN_CUSTOM_NETWORK_OPTION", "custom network design requests require solution design")
    _prepend_transition(workflow, "HANDLE_SERVICE_CHANGE", "CONFIGURE_SERVICE_ORDER", "eligible service change requires an order configuration step")
    _prepend_transition(workflow, "CAPTURE_COMPLAINT_DETAILS", "CHECK_SERVICE_HEALTH", "service-related complaint requires health diagnostics")
    _prepend_transition(workflow, "CAPTURE_COMPLAINT_DETAILS", "REVIEW_REGULATORY_RISK", "regulatory or unresolved complaints require risk review")
    _prepend_transition(workflow, "CHECK_SERVICE_HEALTH", "CHECK_SERVICE_OUTAGE", "service health review requires outage correlation")
    _prepend_transition(workflow, "CHECK_SERVICE_OUTAGE", "RUN_LINE_TEST", "service outage or speed complaint requires a line test")
    _prepend_transition(workflow, "SEND_CUSTOMER_CONFIRMATION", "PREPARE_CUSTOMER_SUMMARY", "customer communication should include a structured case summary")
    _prepend_transition(workflow, "LOG_INTERACTION", "RECONCILE_CASE_NOTES", "interaction logging requires case-note reconciliation")


def _add_100_node_states(workflow: dict[str, Any]) -> None:
    add = _state_adder(workflow)
    states = [
        ("CHECK_CUSTOMER_TENURE", "call_tenure_agent", "tenure affects retention and goodwill options"),
        ("REVIEW_PRODUCT_BUNDLE", "call_bundle_agent", "product bundle constraints affect service changes"),
        ("CHECK_DEVICE_COMPATIBILITY", "call_device_agent", "device compatibility affects fulfilment"),
        ("VALIDATE_STATIC_IP_POLICY", "call_static_ip_policy_agent", "static IP requests require policy validation"),
        ("CHECK_NUMBER_PORTING_STATUS", "call_porting_agent", "voice changes require porting-status review"),
        ("CHECK_SIP_CAPACITY", "call_sip_capacity_agent", "SIP capacity requests require capacity planning"),
        ("ASSESS_NETWORK_CAPACITY", "call_network_capacity_agent", "network capacity affects upgrade feasibility"),
        ("CHECK_INSTALLATION_WINDOW", "call_install_window_agent", "installation timing affects relocation handling"),
        ("VALIDATE_SITE_ACCESS", "call_site_access_agent", "technician visits require site-access validation"),
        ("CHECK_FIELD_RESOURCE_AVAILABILITY", "call_field_capacity_agent", "field resource availability affects scheduling"),
        ("CONFIRM_APPOINTMENT_SLOT", "call_slot_confirmation_agent", "appointment slots must be confirmed"),
        ("CHECK_EQUIPMENT_SHIPMENT", "call_equipment_agent", "equipment shipment may be required"),
        ("CREATE_DISPATCH_ORDER", "call_dispatch_order_agent", "field dispatch requires a dispatch order"),
        ("CHECK_OUTAGE_REGION", "call_outage_region_agent", "regional outage affects technical handling"),
        ("CHECK_MAINTENANCE_WINDOW", "call_maintenance_agent", "planned maintenance may explain service impact"),
        ("CHECK_MAJOR_INCIDENT", "call_incident_agent", "major incident status affects escalation"),
        ("RUN_MODEM_DIAGNOSTICS", "call_modem_diagnostics_agent", "modem diagnostics refine fault handling"),
        ("RUN_ACCESS_NETWORK_TEST", "call_access_test_agent", "access network testing refines fault handling"),
        ("CHECK_PREVIOUS_FAULTS", "call_fault_history_agent", "repeat-fault history affects escalation"),
        ("ASSESS_CHRONIC_FAULT_RISK", "call_chronic_fault_agent", "chronic fault risk affects complaint escalation"),
        ("VERIFY_INVOICE_LINE_ITEMS", "call_invoice_line_agent", "invoice line items must be checked for disputes"),
        ("CHECK_TAX_AND_FEES", "call_tax_fee_agent", "tax and fee questions require specialist checks"),
        ("VALIDATE_CREDIT_POLICY", "call_credit_policy_agent", "credits must comply with policy"),
        ("CALCULATE_GOODWILL_CREDIT", "call_goodwill_agent", "goodwill credit may resolve dissatisfaction"),
        ("CHECK_REFUND_METHOD", "call_refund_method_agent", "refund method must be available"),
        ("CREATE_BILLING_CASE", "call_billing_case_agent", "billing investigation requires a case"),
        ("CHECK_COLLECTIONS_STATUS", "call_collections_agent", "collections state affects payment support"),
        ("VALIDATE_DIRECT_DEBIT_STATUS", "call_direct_debit_agent", "direct debit status affects payment-method answers"),
        ("CHECK_CONTRACT_END_DATE", "call_contract_end_agent", "contract end date affects cancellation and changes"),
        ("ASSESS_EARLY_TERMINATION_FEE", "call_termination_fee_agent", "termination fees affect cancellation handling"),
        ("EVALUATE_CANCELLATION_ELIGIBILITY", "call_cancellation_agent", "cancellation requests require eligibility checks"),
        ("REVIEW_SAVE_OFFER_POLICY", "call_save_policy_agent", "save offers must comply with policy"),
        ("GENERATE_RETENTION_OPTIONS", "call_retention_options_agent", "retention options are generated for eligible customers"),
        ("RECORD_RETENTION_OUTCOME", "call_retention_outcome_agent", "retention outcome must be recorded"),
        ("CHECK_REGULATORY_DEADLINE", "call_regulatory_deadline_agent", "regulatory complaints have response deadlines"),
        ("PREPARE_REGULATORY_BRIEF", "call_regulatory_brief_agent", "formal complaint needs a regulatory brief"),
        ("ASSIGN_COMPLAINT_OWNER", "call_complaint_owner_agent", "complaint ownership must be assigned"),
        ("CHECK_EXECUTIVE_ESCALATION", "call_executive_escalation_agent", "executive escalation changes handling path"),
        ("REVIEW_LEGAL_HOLD", "call_legal_hold_agent", "legal hold affects record handling"),
        ("REDACT_SENSITIVE_NOTES", "call_redaction_agent", "sensitive notes must be redacted"),
        ("VALIDATE_COMMUNICATION_CHANNEL", "call_channel_validation_agent", "communication channel must be validated"),
        ("SEND_SMS_NOTIFICATION", "call_sms_agent", "SMS notification may be required"),
        ("SEND_EMAIL_NOTIFICATION", "call_email_agent", "email notification may be required"),
        ("CREATE_CUSTOMER_PORTAL_UPDATE", "call_portal_update_agent", "portal update records next steps"),
        ("CHECK_NOTIFICATION_DELIVERY", "call_delivery_agent", "delivery status must be confirmed"),
        ("ARCHIVE_WORKFLOW_RECORD", "call_archive_agent", "workflow record must be archived"),
        ("UPDATE_ANALYTICS_EVENTS", "call_analytics_agent", "analytics event should be emitted"),
        ("REVIEW_PROCESS_EXCEPTION", "call_exception_review_agent", "process exception requires review"),
        ("REQUEST_SUPERVISOR_SIGNOFF", "call_supervisor_agent", "supervisor signoff is required"),
        ("FINAL_GOVERNANCE_CHECK", "call_governance_agent", "governance checks must pass before closure"),
    ]
    for state_id, action, condition in states:
        add(state_id, action, [("REQUEST_MANUAL_APPROVAL", condition), ("CREATE_SUPPORT_TICKET", "manual follow-up is required if automated handling cannot complete")])

    _replace_transitions(workflow, "CHECK_FRAUD_SIGNALS", [("RETRIEVE_CUSTOMER_PROFILE", "fraud checks are clear"), ("REVIEW_SECURITY_OR_PRIVACY_RISK", "fraud indicators are present")])
    _replace_transitions(workflow, "VALIDATE_DIRECT_DEBIT_STATUS", [("RETRIEVE_BILLING_PROFILE", "direct debit status is validated"), ("CREATE_SUPPORT_TICKET", "direct debit record requires correction")])
    _replace_transitions(workflow, "VERIFY_INVOICE_LINE_ITEMS", [("ASSESS_BILLING_DISPUTE", "invoice line items confirm the dispute"), ("SEND_CUSTOMER_CONFIRMATION", "invoice line items resolve the question")])
    _replace_transitions(workflow, "VALIDATE_CREDIT_POLICY", [("HANDLE_COMPLAINT", "credit policy permits an offer"), ("REQUEST_MANUAL_APPROVAL", "credit requires approval")])
    _replace_transitions(workflow, "CHECK_CONTRACT_END_DATE", [("VERIFY_SERVICE_INVENTORY", "contract dates allow service-change checks"), ("CHECK_SITE_SERVICEABILITY", "relocation requires site serviceability review"), ("DESIGN_CUSTOM_NETWORK_OPTION", "custom design requires solution design"), ("REQUEST_MANUAL_APPROVAL", "contract dates require commercial approval")])
    _replace_transitions(workflow, "ASSESS_NETWORK_CAPACITY", [("ASSESS_CHANGE_ELIGIBILITY", "network capacity supports eligibility assessment"), ("REQUEST_MANUAL_APPROVAL", "capacity constraint requires approval")])
    _replace_transitions(workflow, "CHECK_INSTALLATION_WINDOW", [("SCHEDULE_TECHNICIAN", "installation window requires a technician appointment"), ("HANDLE_SERVICE_CHANGE", "installation window allows automated fulfilment")])
    _replace_transitions(workflow, "CHECK_FIELD_RESOURCE_AVAILABILITY", [("CONFIRM_APPOINTMENT_SLOT", "field resource is available"), ("REQUEST_MANUAL_APPROVAL", "field resource constraint requires coordination")])
    _replace_transitions(workflow, "CONFIRM_APPOINTMENT_SLOT", [("CREATE_DISPATCH_ORDER", "appointment slot is confirmed"), ("REQUEST_MANUAL_APPROVAL", "appointment slot cannot be confirmed")])
    _replace_transitions(workflow, "CREATE_DISPATCH_ORDER", [("CREATE_SUPPORT_TICKET", "dispatch order is created and must be tracked"), ("REQUEST_MANUAL_APPROVAL", "dispatch order requires approval")])
    _replace_transitions(workflow, "CHECK_OUTAGE_REGION", [("RUN_LINE_TEST", "regional status does not fully explain the service issue"), ("HANDLE_COMPLAINT", "regional outage explains the issue and can be communicated")])
    _replace_transitions(workflow, "RUN_MODEM_DIAGNOSTICS", [("SCHEDULE_TECHNICIAN", "modem diagnostics indicate a field fault"), ("HANDLE_COMPLAINT", "modem diagnostics support non-field resolution")])
    _replace_transitions(workflow, "CHECK_REGULATORY_DEADLINE", [("PREPARE_REGULATORY_BRIEF", "regulatory deadline requires a formal brief"), ("REQUEST_MANUAL_APPROVAL", "deadline risk requires immediate escalation")])
    _replace_transitions(workflow, "PREPARE_REGULATORY_BRIEF", [("REQUEST_MANUAL_APPROVAL", "formal complaint brief requires owner approval"), ("ASSIGN_COMPLAINT_OWNER", "complaint owner can be assigned")])
    _replace_transitions(workflow, "VALIDATE_COMMUNICATION_CHANNEL", [("SEND_EMAIL_NOTIFICATION", "email is the preferred valid channel"), ("SEND_SMS_NOTIFICATION", "SMS is the preferred valid channel")])
    _replace_transitions(workflow, "SEND_EMAIL_NOTIFICATION", [("CHECK_NOTIFICATION_DELIVERY", "email notification is sent"), ("CREATE_SUPPORT_TICKET", "email notification fails")])
    _replace_transitions(workflow, "SEND_SMS_NOTIFICATION", [("CHECK_NOTIFICATION_DELIVERY", "SMS notification is sent"), ("CREATE_SUPPORT_TICKET", "SMS notification fails")])
    _replace_transitions(workflow, "CHECK_NOTIFICATION_DELIVERY", [("LOG_INTERACTION", "delivery is confirmed and can be logged"), ("CREATE_SUPPORT_TICKET", "delivery failure requires follow-up")])
    _replace_transitions(workflow, "ARCHIVE_WORKFLOW_RECORD", [("UPDATE_ANALYTICS_EVENTS", "workflow record is archived"), ("CREATE_SUPPORT_TICKET", "archive failure requires follow-up")])
    _replace_transitions(workflow, "UPDATE_ANALYTICS_EVENTS", [("CLOSE_REQUEST", "analytics event is emitted"), ("CREATE_SUPPORT_TICKET", "analytics event failure requires follow-up")])
    _replace_transitions(workflow, "FINAL_GOVERNANCE_CHECK", [("REQUEST_COMPLETE", "final governance checks pass"), ("REQUEST_ESCALATED", "governance check identifies unresolved risk")])

    _prepend_transition(workflow, "VERIFY_CUSTOMER_IDENTITY", "CHECK_FRAUD_SIGNALS", "identity verification includes fraud-signal screening")
    _prepend_transition(workflow, "CHECK_ACCOUNT_STATUS", "VALIDATE_DIRECT_DEBIT_STATUS", "payment-method requests require direct-debit validation")
    _prepend_transition(workflow, "VALIDATE_PAYMENT_HISTORY", "VERIFY_INVOICE_LINE_ITEMS", "invoice line items must be verified for payment disputes")
    _prepend_transition(workflow, "REVIEW_BILLING_ADJUSTMENT", "VALIDATE_CREDIT_POLICY", "billing adjustment must satisfy credit policy")
    _prepend_transition(workflow, "CHECK_CONTRACT_TERMS", "CHECK_CONTRACT_END_DATE", "contract end date affects change eligibility")
    _prepend_transition(workflow, "VERIFY_SERVICE_INVENTORY", "ASSESS_NETWORK_CAPACITY", "network capacity affects service modification feasibility")
    _prepend_transition(workflow, "CHECK_SITE_SERVICEABILITY", "CHECK_INSTALLATION_WINDOW", "relocation requires installation-window assessment")
    _prepend_transition(workflow, "SCHEDULE_TECHNICIAN", "CHECK_FIELD_RESOURCE_AVAILABILITY", "technician scheduling requires field-resource availability")
    _prepend_transition(workflow, "CHECK_SERVICE_HEALTH", "CHECK_OUTAGE_REGION", "service health checks include regional outage review")
    _prepend_transition(workflow, "RUN_LINE_TEST", "RUN_MODEM_DIAGNOSTICS", "line tests are followed by modem diagnostics when needed")
    _prepend_transition(workflow, "REVIEW_REGULATORY_RISK", "CHECK_REGULATORY_DEADLINE", "regulatory risk requires deadline assessment")
    _prepend_transition(workflow, "CHECK_RETENTION_ELIGIBILITY", "CHECK_CUSTOMER_TENURE", "retention eligibility considers customer tenure")
    _prepend_transition(workflow, "SEND_CUSTOMER_CONFIRMATION", "VALIDATE_COMMUNICATION_CHANNEL", "customer confirmation requires channel validation")
    _prepend_transition(workflow, "LOG_INTERACTION", "ARCHIVE_WORKFLOW_RECORD", "logged interactions must be archived")
    _prepend_transition(workflow, "CLOSE_REQUEST", "FINAL_GOVERNANCE_CHECK", "closure requires final governance check")


def _state_adder(workflow: dict[str, Any]):
    def add(state_id: str, action: str, transitions: list[tuple[str, str]]) -> None:
        workflow["states"][state_id] = {
            "action": action,
            "action_params": {
                "customer_id": "",
                "request_context": "",
            },
            "transitions": [{"to": target, "condition": condition} for target, condition in transitions],
        }

    return add


def _prepend_transition(workflow: dict[str, Any], state_id: str, target: str, condition: str) -> None:
    transitions = workflow["states"][state_id]["transitions"]
    if all(transition["to"] != target for transition in transitions):
        transitions.insert(0, {"to": target, "condition": condition})


def _replace_transitions(workflow: dict[str, Any], state_id: str, transitions: list[tuple[str, str]]) -> None:
    workflow["states"][state_id]["transitions"] = [
        {"to": target, "condition": condition} for target, condition in transitions
    ]
